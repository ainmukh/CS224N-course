import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size,
                                           padding_idx=self.target_vocab.char_pad)

    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size)
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters.
         A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters.
         A tuple of two tensors of shape (1, batch, hidden_size)
        """
        # print('x shape =', input.shape)
        embeddings = self.decoderCharEmb(input)  # (length, batch_size, char_emb)
        # print('x_emb shape =', embeddings.shape)
        hidden_state, dec_hidden = self.charDecoder(embeddings, dec_hidden)
        # print('hidden shape =', hidden_state.shape)
        scores = self.char_output_projection(hidden_state)
        # print('scores shape =', scores.shape)
        # print('self.vocab_size =', len(self.target_vocab.char2id))

        return scores, dec_hidden

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size).
         Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM,
         obtained from the output of the word-level decoder.
         A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor),
         computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        scores, _ = self.forward(char_sequence[:-1], dec_hidden)  # (length, batch_size, self.vocab_size)
        scores = scores.permute(1, 2, 0).contiguous()  # (batch_size, self.vocab_size, length)

        loss_char_dec = nn.CrossEntropyLoss(reduction='sum', ignore_index=self.target_vocab.char_pad)

        # input = [START, x1, x2, ..., xn]
        # probabilities = [x1, x2, ..., xn, END]
        return loss_char_dec(scores, char_sequence[1:].permute(1, 0).contiguous())

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM,
         a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """
        batch_size = initialStates[0].size(1)
        dec_hidden = initialStates

        # в этих переменных числа
        START, END = self.target_vocab.start_of_word, self.target_vocab.end_of_word
        output_word = [''] * batch_size
        # print('output_word is', output_word)
        current_char = torch.tensor([START] * batch_size, device=device)  # (1, batch_size)
        # print('current_char shape =', current_char.size())

        for _ in range(max_length):
            scores, dec_hidden = self.forward(current_char.unsqueeze(0), dec_hidden)
            # print('scores shape =', scores.size())
            softmax = nn.Softmax(1)
            probabilities = softmax(scores.squeeze(0))
            # print('probabilities shape =', probabilities.size())
            current_char = torch.argmax(probabilities, dim=1)
            # print('current_char size =', current_char.size())
            for i in range(batch_size):
                output_word[i] += self.target_vocab.id2char[current_char[i].item()]

        for i in range(batch_size):
            end_idx = output_word[i].find(self.target_vocab.id2char[END])
            output_word[i] = output_word[i][:end_idx] if end_idx != -1 else output_word[i]

        return output_word
