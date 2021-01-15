import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
nltk.download('punkt')


def read_corpus(path: str, source: str) -> List[List]:
    data = []
    for line in open(path):
        sentence = nltk.word_tokenize(line)
        if source == 'tgt':
            sentence = ['<s>'] + sentence + ['</s>']
        data.append(sentence)

    return data


# A5
# копипаст
def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and longest words in all sentences.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """

    sents_padded = []
    max_word_length = max(len(w) for s in sents for w in s)
    max_sent_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for k in range(batch_size):
        sentence = sents[k]
        sent_padded = []

        for w in sentence:
            data = [c for c in w] + [char_pad_token for _ in range(max_word_length-len(w))]
            if len(data) > max_word_length:
                data = data[:max_word_length]
            sent_padded.append(data)

        sent_padded = sent_padded[:max_sent_len] + [[char_pad_token]*max_word_length] * max(0, max_sent_len - len(sent_padded))
        sents_padded.append(sent_padded)
    return sents_padded
# /копипаст


def batch_iter(data: List[Tuple], batch_size: int, shuffle: bool = False):
    batch_num = math.ceil(len(data) / batch_size)  # количество батчей
    indices = list(range(len(data)))

    if shuffle:
        np.random.shuffle(indices)

    for i in range(batch_num):
        indices_i = indices[batch_size * i:batch_size * (i + 1)]
        sents = [data[idx] for idx in indices_i]

        sents = sorted(sents, key=lambda s: len(s[0]), reverse=True)

        src_sents = [s[0] for s in sents]
        tgt_sents = [s[1] for s in sents]

        yield src_sents, tgt_sents
