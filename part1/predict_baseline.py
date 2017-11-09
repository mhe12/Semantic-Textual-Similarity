import pdb
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
import os


class Data(object):
    def __init__(self):
        self.dictionary = set()
        self.st_pair = []
        self.word2vec = {}
        self.notfind = []
        self.cos_dist = []
        self.stopwords = set()

class Similarity(object):
    def __init__(self, input_file, output_file, embedding, stopwords_file):
        self.data = Data()
        self.input_file = input_file
        self.output_file = output_file
        self.embedding = embedding
        self.stopwords_file = stopwords_file

    def evaluate(self):
        self.read_input()
        self.read_stopwords(self.stopwords_file)
        self.get_embedding(self.embedding[0], self.data.dictionary)
        self.get_embedding(self.embedding[1], self.data.notfind, False)
        self.compute_cos_dist()
        self.write_similarity()

    def read_stopwords(self, stopwords_file):
        print 'reading stopwords ...'
        with open(stopwords_file, 'r') as infile:
            self.data.stopwords = set([line.lower().decode('utf-8')[:-1] for line in infile])

    def read_input(self):
        print 'reading input data ...'
        with open(self.input_file, 'r') as infile:
            for line in infile:
                sentence = line.strip().split('\t')
                st1 = word_tokenize(sentence[0].lower().decode('utf8').replace('" ','"'))
                st2 = word_tokenize(sentence[1].lower().decode('utf8').replace('" ','"'))
                self.data.st_pair.append((st1,st2))
                words = set(st1) | set(st2)
                self.data.dictionary = self.data.dictionary | words

    def get_embedding(self, vector_lib, find_set, decode = True):
        print 'finding the word2vec embeddings in ', vector_lib.split('/')[-1],  '...'
        visited = set()
        with open(vector_lib, 'r') as infile:
            for line in infile:
                line = line.split(' ')
                if decode: 
                    word = line[0].lower().decode('utf8')
                else: 
                    word = line[0].lower()
                if word in find_set:
                    vec = np.array(map(float, line[1:]))
                    self.data.word2vec[word] = vec
                    visited.add(word)
            self.data.notfind = find_set - visited


    def compute_cos_dist(self):
        print 'computing cosine similarity ...'
        for i in xrange(len(self.data.st_pair)):
            st1_vec = sum(self.data.word2vec[w.lower()] for w in self.data.st_pair[i][0] if w.lower() in self.data.word2vec.keys() and w.lower() not in self.data.stopwords)
            st2_vec = sum(self.data.word2vec[w.lower()] for w in self.data.st_pair[i][1] if w.lower() in self.data.word2vec.keys() and w.lower() not in self.data.stopwords)
            result = cosine_similarity(st1_vec.reshape(1,-1), st2_vec.reshape(1,-1))[0][0] * 5
            self.data.cos_dist.append(result)


    def write_similarity(self):
        print 'writing result ...'
        try:
            os.remove(self.output_file)
        except OSError:
            pass
        with open(self.output_file,'w') as out:
            for i in self.data.cos_dist:
                out.write("%s\n" % i)


if __name__ == "__main__":
    args = sys.argv
    print args
    embedding = ['../datasets+scoring_script/paragram-phrase-XXL.txt','../datasets+scoring_script/paragram_300_sl999.txt']
    stopwords_file = 'stopwords.txt'
    sim = Similarity(args[-2], args[-1], embedding, stopwords_file)
    sim.evaluate()

