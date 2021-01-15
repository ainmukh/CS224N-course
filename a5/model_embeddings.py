import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import VocabEntry


class CNN(nn.Module):
    def __init__(self, char_embed_size, word_embed_size: int, kernel_size: int, padding: int):
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=char_embed_size,
                              out_channels=word_embed_size,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, reshaped):
        # reshaped: (batch_size * max_sentence_length, charm_embed_size, max_word_length)
        # conv: (batch_size * max_sentence_length, word_embed_size, max_word_length + 2 * padding - k + 1)
        conv = self.conv(reshaped)
        conv_out = F.max_pool1d(F.relu(conv), kernel_size=conv.shape[-1]).squeeze(-1)
        return conv_out


class Highway(nn.Module):
    def __init__(self, word_embed_size: int):
        super(Highway, self).__init__()

        self.w_proj = nn.Linear(word_embed_size, word_embed_size)
        self.w_gate = nn.Linear(word_embed_size, word_embed_size)

    def forward(self, conv_out):
        # conv_out: (batch_size * max_sentence_length, word_embed_size)
        proj = F.relu(self.w_proj(conv_out))
        gate = torch.sigmoid(self.w_gate(conv_out))

        highway = gate * proj + (1 - gate) * conv_out

        return highway


# в A5 для каждого языка свой имбединг
# тут имбединг не тупа подглядеть в табличку а еще поколдовать со слоями
class ModelEmbeddings(nn.Module):

    def __init__(self,
                 word_embed_size: int,
                 vocab: VocabEntry,
                 char_embed_size: int = 50,
                 kernel_size: int = 5,
                 padding: int = 1,
                 dropout_rate: float = 0.3):
        super(ModelEmbeddings, self).__init__()

        self.vocab = vocab  # A5
        self.word_embed_size = word_embed_size
        self.char_embed_size = char_embed_size  # A5

        # A simple lookup table that stores embeddings of a fixed dictionary and size.
        # This module is often used to store word embeddings and retrieve them using indices.
        # The input to the module is a list of indices, and the output is the corresponding word embeddings.
        # padding_idx (int, optional) – If given,
        # pads the output with the embedding vector at padding_idx (initialized to zeros)
        # whenever it encounters the index.
        self.embedding = nn.Embedding(num_embeddings=len(vocab.char2id),
                                      embedding_dim=char_embed_size,
                                      padding_idx=vocab.char_pad)
        self.cnn = CNN(self.char_embed_size, self.word_embed_size, kernel_size, padding)
        self.highway = Highway(self.word_embed_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, padded):
        sentence_length, batch_size, max_word_length = padded.size()
        # padded -> word_emb
        # padded: (sentence_length, batch, max_word_length) ->
        # emb: (sentence_length, batch, max_word_length, char_embed_size)
        emb = self.embedding(padded)
        # reshaped: (batch_size * max_sentence_length, charm_embed_size, max_word_length)
        reshaped = emb.permute(0, 1, 3, 2).reshape(
            sentence_length * batch_size, self.char_embed_size, -1
        ).contiguous()

        # conv_out: (batch_size * max_sentence_length, word_embed_size)
        conv_out = self.cnn(reshaped)

        # highway: (batch_size * max_sentence_length, word_embed_size)
        highway = self.highway(conv_out)

        # word_emb: (batch_size * max_sentence_length, word_embed_size)
        word_emb = self.dropout(highway).reshape(sentence_length, batch_size, self.word_embed_size)

        return word_emb
