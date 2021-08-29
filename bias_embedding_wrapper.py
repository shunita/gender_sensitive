import numpy as np
from glove import Glove, Corpus
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine as cos_dist


class EmbedModelWrapper(object):

    def __init__(self, use_glove, min_count, vector_size, iterations, window):
        self.glove_model = None
        self.w2v_model = None
        self.use_glove = use_glove
        self.min_count = min_count
        self.vector_size = vector_size
        self.iterations = iterations
        self.window = window

    def train(self, tokenized_abstracts):
        # @param tokenized_abstracts - a list of lists of strings
        # Based on this: https://towardsdatascience.com/a-beginners-guide-to-word-embedding-with-gensim-word2vec-model-5970fa56cc92
        print("training model")
        self.w2v_model = Word2Vec(
            tokenized_abstracts, min_count=self.min_count, size=self.vector_size,
            workers=3, window=self.window, sg=0, iter=self.iterations)
        i = 0
        word_to_index = {}
        for w in self.w2v_model.wv.vocab:
            word_to_index[w] = i
            i += 1
        if self.use_glove:
            corpus_model = Corpus(word_to_index)
            corpus_model.fit(tokenized_abstracts, window=self.window, ignore_missing=True)
            print('Dict size: {}'.format(len(corpus_model.dictionary)))
            print('Collocations: {}'.format(corpus_model.matrix.nnz))
            glove = Glove(no_components=self.vector_size, learning_rate=0.05)
            glove.fit(corpus_model.matrix, epochs=self.iterations,
                      no_threads=3, verbose=True)
            glove.add_dictionary(corpus_model.dictionary)
            self.glove_model = glove

            w2v_vocab = set(list(self.w2v_model.wv.vocab.keys()))
            glove_vocab = set(list(self.glove_model.dictionary.keys()))

            print("{} in w2v. {} in glove.\n{} in w2v but not in glove.\n{} in glove but not in w2v.".format(
                len(w2v_vocab),
                len(glove_vocab),
                len(w2v_vocab.difference(glove_vocab)),
                len(glove_vocab.difference(w2v_vocab))
            ))

    def save(self, filename):
        # save both models
        self.w2v_model.save(filename + "_w2v")
        if self.use_glove and self.glove_model is not None:
            self.glove_model.save(filename + "_glove")

    def load(self, filename):
        # load both models
        self.w2v_model = Word2Vec.load(filename + "_w2v")
        if self.use_glove:
            self.glove_model = Glove.load(filename + "_glove")

    def similarity(self, w1, w2):
        if self.use_glove:
            word_idx1 = self.glove_model.dictionary[w1]
            word_idx2 = self.glove_model.dictionary[w2]
            v1 = self.glove_model.word_vectors[word_idx1]
            v2 = self.glove_model.word_vectors[word_idx2]
            return 1 - cos_dist(v1, v2)
        return self.w2v_model.wv.similarity(w1, w2)

    def get_vocab(self):
        if self.use_glove:
            return self.glove_model.dictionary
        return self.w2v_model.wv.vocab

    def get_vocab_list(self):
        if self.use_glove:
            return list(self.glove_model.dictionary.keys())
        return list(self.w2v_model.wv.vocab.keys())

    def get_count(self, w):
        if w in self.w2v_model.wv.vocab:
            return self.w2v_model.wv.vocab[w].count
        return 0

    def most_similar(self, word, topn=5):
        top = []
        if self.use_glove:
            if word in self.glove_model.dictionary:
                top = self.glove_model.most_similar(word=word, number=topn)
            return top
        # Or use w2v
        if word in self.w2v_model.wv.vocab:
            top = self.w2v_model.most_similar(positive=word, topn=topn)
        return top

    def get_word_vectors_as_matrix_rows(self, index_to_word):
        """index_to_word - a list of words in the required order"""
        vectors = []
        for w in index_to_word:
            if self.use_glove:
                glove_idx = self.glove_model.dictionary[w]
                vectors.append(self.glove_model.word_vectors[glove_idx])
            else:
                vectors.append(self.w2v_model.wv[w])
        return np.array(vectors)

    def export_embeddings(self, output_file):
        """model is an instance of EmbedModelWrapper"""
        vocab = self.get_vocab_list()
        matrix = self.get_word_vectors_as_matrix_rows(vocab)
        out = open(output_file, "w")
        for i in range(len(vocab)):
            vector_as_str = ",".join([str(x) for x in matrix[i]])
            out.write("{}\t{}\n".format(vocab[i], vector_as_str))


class ModelMerger(EmbedModelWrapper):
    def __init__(self, models):
        self.index_to_word, self.word_to_index, self.counter = self.merge_vocabs(models)
        self.sim = self.calc_similarity_matrix(models)

    def merge_vocabs(self, models):
        word_set = set(models[0].get_vocab())
        for model in models[1:]:
            before = len(word_set)
            word_set = word_set.union(set(model.get_vocab()))
            after = len(word_set)
            if after > before:
                print("Warning: found misatching vocabs in merger.")
        index_to_word = list(word_set)
        word_to_index = {index_to_word[index]: index for index in range(len(index_to_word))}
        counter = {index_to_word[index]: models[0].get_count(index_to_word[index]) for index in range(len(index_to_word))}
        return index_to_word, word_to_index, counter

    def calc_similarity_matrix(self, models):
        matrices = []
        for model in models:
            m1 = model.get_word_vectors_as_matrix_rows(self.index_to_word)
            m2 = m1 / np.linalg.norm(m1, axis=1)[:, np.newaxis]
            matrices.append(np.dot(m2, m2.transpose()))
        return np.mean(np.array(matrices), axis=0)

    def most_similar(self, w, topn=5):
        if w in self.word_to_index:
            sim_to_w = self.sim[self.word_to_index[w]]
            word_ids = np.argsort(-sim_to_w)
            # the first will likely be the original word
            neighbors = [(self.index_to_word[x], sim_to_w[x]) for x in word_ids[:topn+1]]
            if neighbors[0][0] == w:
                neighbors = neighbors[1:]
            return neighbors
        return []

    def get_vocab(self):
        return self.word_to_index

    def get_vocab_list(self):
        return self.index_to_word

    def similarity(self, w1, w2):
        return self.sim[self.word_to_index[w1], self.word_to_index[w2]]

    def get_count(self, w):
        return self.counter.get(w, 0)
