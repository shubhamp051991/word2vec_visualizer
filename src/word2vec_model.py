import numpy as np
from gensim.models import Word2Vec

class Word2VecModel:
    def __init__(self, sentences, architecture='cbow', vector_size=100, window=5, min_count=1):
        self.sentences = [sentence.split() for sentence in sentences]
        self.architecture = 0 if architecture == 'cbow' else 1  # 0 for CBOW, 1 for Skip-gram
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.model = None

    def train(self):
        self.model = Word2Vec(sentences=self.sentences,
                              vector_size=self.vector_size,
                              window=self.window,
                              min_count=self.min_count,
                              sg=self.architecture)

    def get_word_vectors(self):
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        return {word: self.model.wv[word] for word in self.model.wv.key_to_index}

    def vector_arithmetic(self, word1, word2, word3):
        if self.model is None:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        result = self.model.wv.most_similar(positive=[word1, word3], negative=[word2], topn=1)
        return result[0][0]

class CBOW(Word2VecModel):
    def __init__(self, sentences, **kwargs):
        super().__init__(sentences, architecture='cbow', **kwargs)

class SkipGram(Word2VecModel):
    def __init__(self, sentences, **kwargs):
        super().__init__(sentences, architecture='skipgram', **kwargs)
