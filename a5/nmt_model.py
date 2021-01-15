from collections import namedtuple
import sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F

from model_embeddings import ModelEmbeddings
from char_decoder import CharDecoder
from vocab import Vocab

import random

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    def __init__(self,
                 word_embed_size: int,
                 hidden_size: int,
                 vocab: Vocab,
                 dropout_rate: float = 0.2,
                 use_char_decoder: bool = True):
        """
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(NMT, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        #         self.model_embeddings = ModelEmbeddings(word_embed_size, vocab)
        # A5
        # для каждого языка свой имбеддинг
        self.model_embeddings_source = ModelEmbeddings(word_embed_size, vocab.src)
        self.model_embeddings_target = ModelEmbeddings(word_embed_size, vocab.tgt)
        self.encoder = nn.LSTM(word_embed_size, hidden_size, bidirectional=True)
        self.hidden_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.cell_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.decoder = nn.LSTMCell(word_embed_size + hidden_size, hidden_size)
        self.att_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)

        if use_char_decoder:
            self.charDecoder = CharDecoder(hidden_size, target_vocab=vocab.tgt)
        else:
            self.charDecoder = None

    @property
    def device(self) -> torch.device:
        # the learnable weights of the module of shape (src_vocab_size, word_embed_size) initialized from N(0,1)
        #         return self.model_embeddings.source.weight.device
        # A5
        return self.att_projection.weight.device

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        source_lengths = [len(sentence) for sentence in source]

        #         source_padded = self.vocab.src.to_input_tensor(source, device=self.device)
        # (tgt_sentence_length, batch)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)
        # A5
        source_padded_chars = self.vocab.src.to_input_tensor_char(source, device=self.device)
        # (max_sentence_length, batch_size, max_word_length)
        target_padded_chars = self.vocab.tgt.to_input_tensor_char(target, device=self.device)

        enc_hiddens, dec_init_state = self.encode(source_padded_chars, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        # (tgt_sentence_len, batch, hidden)
        combined_outputs = self.decode(enc_hiddens, dec_init_state, target_padded_chars, enc_masks)

        # predicted log(probability)
        # (tgt_sentence_len, batch, tgt_vocab_size)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)
        # slice P with indices of gold words
        # (tgt_sentence_len, batch, tgt_vocab_size)[(tgt_sentence_length, batch, 1)] = (tgt_sentence_len, batch)
        # чтобы не слайсить из словаря <START> вырезаем его [1:]
        target_gold_words_log_prob = torch.gather(P, dim=2, index=target_padded[1:].unsqueeze(-1)).squeeze(-1)
        # zero out probabilities for <pad> tokens
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()
        target_gold_words_log_prob = target_gold_words_log_prob * target_masks[1:]
        scores = target_gold_words_log_prob.sum()

        # A5
        # копипаст
        if self.charDecoder is not None:
            max_word_len = target_padded_chars.shape[-1]
            target_chars = target_padded_chars[1:].view(-1, max_word_len)
            target_outputs = combined_outputs.view(-1, 256)

            target_chars_oov = target_chars  # torch.index_select(target_chars, dim=0, index=oovIndices)
            rnn_states_oov = target_outputs  # torch.index_select(target_outputs, dim=0, index=oovIndices)
            oovs_losses = self.charDecoder.train_forward(target_chars_oov.t().contiguous(),
                                                         (rnn_states_oov.unsqueeze(0), rnn_states_oov.unsqueeze(0)))
            scores = scores - oovs_losses

        return scores

    def encode(self, source_padded: torch.Tensor, source_lengths: List[int]) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # GET WORD EMBEDDINGS
        # sentences are lists of indices of words
        # (src_sentence_length, batch) –> (src_sentence_length, batch, word_emb_size)
        source_emb = self.model_embeddings_source(source_padded)
        # packs a Tensor containing padded sequences of variable length.
        source_packed = nn.utils.rnn.pack_padded_sequence(source_emb, source_lengths)

        # GET ALL HIDDEN STATES AND FINAL HIDDEN STATE, CELL STATE
        enc_hiddens, (hidden_state, cell_state) = self.encoder(source_packed)
        # Pads a packed batch of variable length sequences.
        # It is an inverse operation to pack_padded_sequence().
        # The returned Tensor’s data will be of size T x B x *, where T is the length of the longest
        # (batch, src_sentence_length, hidden_size)
        enc_hiddens, lens_unpacked = nn.utils.rnn.pad_packed_sequence(enc_hiddens, batch_first=True)

        # INITIALIZE DECODER INITIAL STATES
        # first, concatenate hidden_forward and hidden_backward
        # (forward + backward = 2, batch, hidden_size) –> (batch, 2 * hidden_size)
        # первый для бэкварда – последний для форварда, последний для форварда – первый для бэкварда
        # то есть это последние хидден стейты, t = seq_len
        hidden_concat = torch.cat((hidden_state[0], hidden_state[1]), dim=1)
        dec_init_hidden = self.hidden_projection(hidden_concat)

        cell_concat = torch.cat((cell_state[0], cell_state[1]), dim=1)
        dec_init_cell = self.cell_projection(cell_concat)

        return enc_hiddens, (dec_init_hidden, dec_init_cell)

    def decode(self,
               enc_hiddens: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor],
               target_padded: torch.Tensor,
               enc_masks: torch.Tensor) -> torch.Tensor:

        hidden_state, cell_state = dec_init_state

        # (batch, src_sentence_length, 2 * hidden_size) –> (batch, src_sentence_length, hidden_size)
        enc_hiddens_proj = self.att_projection(enc_hiddens)

        # у того, что есть результат нашей модели, мы чопаем <END>
        # (tgt_sentence_length, batch) –> (tgt_sentence_length, batch, word_emb_size)
        target_emb = self.model_embeddings_target(target_padded[:-1])  # ???
        # initialize o_0
        # (batch, hidden_size)
        combined_output = torch.zeros(target_emb.size(1), self.hidden_size, device=self.device)

        combined_outputs = []
        # iterate over words in sentences
        for tgt_emb in torch.split(target_emb, 1):
            # [(1, batch, word_emb_size), (batch, hidden_size)]
            tgt_emb = torch.cat((tgt_emb.squeeze(0), combined_output), 1)

            # STEP
            # (batch, hidden_size)
            (hidden_state, cell_state) = self.decoder(tgt_emb, (hidden_state, cell_state))

            # ATTENTION
            # If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
            # (batch, src_sentence_len, hidden) x (batch, hidden, 1) = (batch, src_sentence_length)
            e_t = torch.bmm(enc_hiddens_proj, hidden_state.unsqueeze(2)).squeeze(2)

            # set e_t to -inf where the word is pad in order to make softmax equal 0
            if enc_masks is not None:
                e_t.masked_fill_(enc_masks.bool(), -float('inf'))

            alpha_t = F.softmax(e_t, 1)
            # <(batch, 1, src_sentence_len), (batch, src_sentence_len, 2 * hidden)> = (batch, 2 * hidden)
            a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens).squeeze(1)
            # [(batch, 2 * hidden), (batch, hidden_size)] = (batch, 3 * hidden_size)
            u_t = torch.cat((a_t, hidden_state), 1)
            # (batch, hidden)
            v_t = self.combined_output_projection(u_t)
            combined_output = self.dropout(torch.tanh(v_t))
            # /STEP

            combined_outputs.append(combined_output)

        combined_outputs = torch.stack(combined_outputs)
        return combined_outputs

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for enc_hidden, src_sentence_len in enumerate(source_lengths):
            enc_masks[enc_hidden, src_sentence_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self,
                    src_sent: List[str],
                    beam_size: int = 5,
                    max_decoding_time_step: int = 70) -> List[Hypothesis]:
        # source_padded
        source = self.vocab.src.to_input_tensor_char([src_sent], self.device)

        enc_hiddens, dec_init_state = self.encode(source, [len(src_sent)])
        # enc_masks = self.generate_sent_masks(enc_hiddens, [len(src_sent)]) НУЖНЫ БЫЛИ ЧТОБЫ ЗАКРЫТЬ ПАДДИНГИ
        # combined_outputs = self.decode(enc_hiddens, dec_init_state, target_padded, enc_masks) САМИ ГЕНЕРИМ ТАРГЕТ
        # ДАЛЬШЕ ВИДОИЗМЕНЕННЫЙ ДЕКОД

        hidden_state, cell_state = dec_init_state
        # (batch = 1, src_sentence_length, 2 * hidden_size) –> (batch = 1, src_sentence_length, hidden_size)
        enc_hiddens_proj = self.att_projection(enc_hiddens)

        # initialize o_0
        # (batch = n = 1, hidden_size)
        combined_output = torch.zeros(1, self.hidden_size, device=self.device)
        # combined_outputs = [] НЕ НУЖНЫ ПОТОМУ ЧТО МЫ ПО ХОДУ ДЕЛА ОТБИРАЕМ ЛУЧШИЕ
        # iterate over words in sentences НЕПРАВДА ПОТОМУ ЧТО ИТЕРИРУЕМСЯ ПОКА ЕСТЬ КУДА
        hypotheses = [['<s>']]  # List[Hypothesis.value]
        hypotheses_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)  # Hypothesis.score
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1  # НАЧИНАЕМ С ПЕРВОГО СЛОВА
            n = len(hypotheses)  # нужно нагенерить n * beam_size новых слов

            tgt_emb = self.model_embeddings_target(self.vocab.tgt.to_input_tensor_char(
                list([hypothesis[-1]] for hypothesis in hypotheses), device=self.device
            )).squeeze(0)  # а пахать то будет?))))
            # [(n, word_emb_size), (n, hidden_size)]
            tgt_emb = torch.cat((tgt_emb, combined_output), 1)
            # (batch, hidden_size)
            (hidden_state, cell_state) = self.decoder(tgt_emb, (hidden_state, cell_state))

            # ATTENTION
            # If input is a (b×n×m) tensor, mat2 is a (b×m×p) tensor, out will be a (b×n×p) tensor.
            # (batch, src_sentence_len, hidden) x (batch, hidden, 1) = (batch, src_sentence_length)
            enc_hiddens_proj_n = enc_hiddens_proj.expand(n, enc_hiddens_proj.size(1), enc_hiddens_proj.size(2))
            e_t = torch.bmm(enc_hiddens_proj_n, hidden_state.unsqueeze(2)).squeeze(2)

            alpha_t = F.softmax(e_t, 1)
            # <(batch, 1, src_sentence_len), (batch, src_sentence_len, 2 * hidden)> = (batch, 2 * hidden)
            enc_hiddens_n = enc_hiddens.expand(n, enc_hiddens.size(1), enc_hiddens.size(2))
            a_t = torch.bmm(alpha_t.unsqueeze(1), enc_hiddens_n).squeeze(1)
            # [(batch, 2 * hidden), (batch, hidden_size)] = (batch, 3 * hidden_size)
            u_t = torch.cat((a_t, hidden_state), 1)
            # (batch, hidden)
            v_t = self.combined_output_projection(u_t)
            combined_output = self.dropout(torch.tanh(v_t))

            # log(probability) of every possible word for every pending hypothesis now
            # (n, tgt_vocab_size)
            P = F.log_softmax(self.target_vocab_projection(combined_output), dim=-1)

            # насколько урезался beam_size
            pending_n = beam_size - len(completed_hypotheses)
            # получаем [скор слов для гипотезы_1, скор слов для гипотезы_2, ..., скор слов для гипотезы_n]
            pending_scores = (hypotheses_scores.unsqueeze(1).expand_as(P) + P).view(-1)
            # выбираем столько слов, сколько нужно сейчас
            # topk возвращает отсорченные в убывающем порядке
            pending_n_scores, pending_n_idx = torch.topk(pending_scores, k=pending_n)

            # находим номера гипотез, которые будут продолжены
            continued_hypotheses = pending_n_idx // len(self.vocab.tgt)
            # номера слов которыми мы заполним гипотезы
            continue_words = pending_n_idx % len(self.vocab.tgt)

            generated_hypotheses = []
            generated_hypotheses_scores = []
            pending_hypotheses_idx = []

            # копипаст
            decoderStatesForUNKsHere = []
            # /копипаст

            for continued_hypothesis, continue_word, continue_score in zip(continued_hypotheses,
                                                                           continue_words,
                                                                           pending_n_scores):
                continued_hypothesis = continued_hypothesis.item()
                continue_word = self.vocab.tgt.id2word[continue_word.item()]
                continue_score = continue_score.item()

                # копипаст
                # initial states of unks
                if continue_word == "<unk>":
                    continue_word = "<unk>" + str(len(decoderStatesForUNKsHere))  # запоминаем номер анка
                    decoderStatesForUNKsHere.append(combined_output[continued_hypothesis])  # для данной гипотезы
                # /копипаст

                hypothesis = hypotheses[continued_hypothesis] + [continue_word]

                # если предсказали завершение предложения:
                if continue_word == '</s>':
                    completed_hypotheses.append(Hypothesis(
                        value=hypothesis[1:-1], score=continue_score
                    ))
                else:
                    generated_hypotheses.append(hypothesis)  # hypotheses на следующей итерации
                    generated_hypotheses_scores.append(continue_score)  # их скоры
                    # чтобы срезать dec_init_state и combined_outputs
                    pending_hypotheses_idx.append(continued_hypothesis)

            if len(decoderStatesForUNKsHere) > 0 and self.charDecoder is not None:  # decode UNKs
                decoderStatesForUNKsHere = torch.stack(decoderStatesForUNKsHere, dim=0)
                # копипаст
                decodedWords = self.charDecoder.decode_greedy(
                    (decoderStatesForUNKsHere.unsqueeze(0), decoderStatesForUNKsHere.unsqueeze(0)),
                    max_length=21,
                    device=self.device
                )
                # /копипаст
                assert len(decodedWords) == decoderStatesForUNKsHere.size()[0], "Incorrect number of decoded words"

                for hypothesis in generated_hypotheses:
                    if hypothesis[-1].startswith("<unk>"):
                        hypothesis[-1] = decodedWords[int(hypothesis[-1][5:])]  # [:-1]

            if len(completed_hypotheses) == beam_size:
                break

            pending_hypotheses_idx = torch.tensor(pending_hypotheses_idx, dtype=torch.long, device=self.device)
            (hidden_state, cell_state) = (hidden_state[pending_hypotheses_idx], cell_state[pending_hypotheses_idx])
            combined_output = combined_output[pending_hypotheses_idx]

            hypotheses = generated_hypotheses
            hypotheses_scores = torch.tensor(generated_hypotheses_scores, dtype=torch.float, device=self.device)

        # если не закончили ни одну гипотезу, то берем первую (и лучшую), срезаем <s>
        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(
                value=hypotheses[0][1:], score=hypotheses_scores[0].item()
            ))

        completed_hypotheses.sort(key=lambda hypothesis: hypothesis.score, reverse=True)

        return completed_hypotheses

    @staticmethod
    def load(model_path: str, no_char_decoder=False):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], use_char_decoder=no_char_decoder, **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(
                word_embed_size=self.model_embeddings_source.word_embed_size,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate
            ),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
