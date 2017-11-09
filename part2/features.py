import sys
import pdb
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from collections import Counter, defaultdict

class Data(object):
    def __init__(self, input_file, decode_way = 'utf-8', filter_stopwords = True):
        self.input_file = input_file
        self.dictionary = set()
        self.st_pair = []
        self.sentence_pair = []
        self.word2vec = {}
        self.notfind = []
        self.text_list = []
        self.golden_score = []
        self.filter_stopwords = filter_stopwords
        self.stopwords = set()
        self.read_input(decode_way)

    def read_stopwords(self, stopwords_file = 'stopwords.txt'):
        print 'reading stopwords ...'
        with open(stopwords_file, 'r') as infile:
            self.stopwords = set([line.lower().decode('utf-8')[:-1] for line in infile])

    def read_input(self, decode_way):
        print 'reading input data ...'
        if self.filter_stopwords:
            self.read_stopwords()
        with open(self.input_file, 'r') as infile:
            for line in infile:
                sentence = line.strip().split('\t')
                self.sentence_pair.append(sentence)
                st1 = [ w for w in word_tokenize(sentence[0].lower().decode(decode_way, errors = 'ignore').replace('" ','"')) if w not in self.stopwords]
                st2 = [ w for w in word_tokenize(sentence[1].lower().decode(decode_way, errors = 'ignore').replace('" ','"')) if w not in self.stopwords]
                self.text_list = self.text_list+st1+st2
                self.st_pair.append((st1,st2))
                words = set(st1) | set(st2)
                self.dictionary = self.dictionary | words

    def read_score(self):
        name = self.input_file.split('.')
        name[-3] = 'gs'
        score_fileName = '.'.join(name)
        with open(score_fileName, 'r') as infile:
            for line in infile:
                self.golden_score.append(float(line))

        return np.array(self.golden_score)


class Feature(object):
    def __init__(self):
        self.data = None
        self.score = []

    def get_data(self, data):
        self.data = data

    def compute_score(self):
        pass


class alignment(Feature):
    def __init__(self):
        super(alignment, self).__init__()
        self.currdir = ''
        self.al = None
        self.errors = []

    def set_align_env(self):
        print 'importing aligner module ...'
        sys.path.append(os.path.join(os.getcwd(), "monolingual-word-aligner"))
        self.currdir = os.getcwd()
        os.chdir('./monolingual-word-aligner')
        import aligner as al
        self.al = al

    def compute_score(self):
        print 'computing alignment score ...'
        self.set_align_env()
        for i in xrange(len(self.data.st_pair)):
            st1, st2 = self.data.st_pair[i]
            try:
                result = self.al.align(st1, st2)
                s = float( 10*len(result[0]) ) / float( (len(st1)+len(st2)) )
                print s
                self.score.append(s )
            except Exception as al_err:
                print 'error'
                self.score.append(2.5)
                self.errors.append(i)
        os.chdir(self.currdir)
        return self.score



class ngram(Feature):
    def __init__(self, n):
        super(ngram, self).__init__()
        self.n = n
        self.lemmatize_pair = []


    def compute_score(self):
        print 'computing n-gram score ...'
        self.word_lemmatize()
        self.compute_ngram()
        return self.score

    def compute_ngram(self):
        for (st1, st2) in self.data.st_pair:
            st1_gram = set([ ''.join(st1[i:i+self.n]) for i in xrange(len(st1)-self.n) ])  # did not consider '.', otherwise +1
            st2_gram = set([ ''.join(st2[i:i+self.n]) for i in xrange(len(st2)-self.n) ])
            s = float( len(st1_gram&st2_gram) ) / float( len(st1_gram|st2_gram) ) * 5
            # print s
            self.score.append(s)
            # pdb.set_trace()

    def word_lemmatize(self):
        print '\t\t lemmatizing input data ...'
        lmtzr = WordNetLemmatizer()

        for p in self.data.st_pair:
            st1 = [ lmtzr.lemmatize(t) for t in p[0] ]
            st2 = [ lmtzr.lemmatize(t) for t in p[1] ]
            self.lemmatize_pair.append((st1,st2))

class nchar(Feature):
    def __init__(self, n):
        super(nchar, self).__init__()
        self.n = n
        self.lemmatize_pair = []

    def compute_score(self):
        print 'computing n-character score ...'
        self.compute_ngram()
        return self.score

    def compute_ngram(self):
        for st in self.data.sentence_pair:
            st1_char = set([ ''.join(st[0][i:i+self.n]) for i in xrange(len(st[0])-self.n) ])  # did not consider '.', otherwise +1
            st2_char = set([ ''.join(st[1][i:i+self.n]) for i in xrange(len(st[1])-self.n) ])
            s = float( len(st1_char&st2_char) ) / float( len(st1_char|st2_char) ) * 5
            # print s
            self.score.append(s)


class cosine(Feature):
    def __init__(self, tf = 0, embedding = ['../datasets+scoring_script/paragram-phrase-XXL.txt','../datasets+scoring_script/paragram_300_sl999.txt']):
        super(cosine, self).__init__()
        self.embedding = embedding
        self.tf = tf
        self.tf_idf = dict()

    def get_tf_idf_score(self):
        doc_freq_per_word = defaultdict(int)
        word_freq = Counter(self.data.text_list)
        highest_freq = float(word_freq.most_common(1)[0][1])
        for (st1, st2) in self.data.st_pair:
            for w in set(st1):
                doc_freq_per_word[w] += 1
            for w in set(st2):
                doc_freq_per_word[w] += 1

        for w in self.data.dictionary:
            self.tf_idf[w] = ( word_freq[w]/highest_freq) * math.log( len(self.data.st_pair)*2 / (doc_freq_per_word[w] + 1)  )


    def compute_score(self):
        print 'computing cosine score ...'
        self.get_embedding(self.embedding[0], self.data.dictionary)
        self.get_embedding(self.embedding[1], self.data.notfind, False)
        if self.tf == 1:
            self.get_tf_idf_score()
        self.compute_cos_dist()
        return self.score


    def get_embedding(self, vector_lib, find_set, decode = True):
        print '\t\t finding the word2vec embeddings in ', vector_lib.split('/')[-1],  '...'
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
            st1_vec = sum(self.data.word2vec[w.lower()]*self.tf_idf.get(w,1) for w in self.data.st_pair[i][0] if w.lower() in self.data.word2vec.keys())
            st2_vec = sum(self.data.word2vec[w.lower()]*self.tf_idf.get(w,1) for w in self.data.st_pair[i][1] if w.lower() in self.data.word2vec.keys())
            s = cosine_similarity(st1_vec.reshape(1,-1), st2_vec.reshape(1,-1))[0][0] * 5
            # print result
            self.score.append(s)



if __name__ == "__main__":
    args = sys.argv
    print args
    # nltk.download('wordnet')
    data = Data('../datasets+scoring_script/train/STS2012-en-train/STS.input.MSRvid.txt')
    golden = data.read_score()
    ngram = ngram(2)
    ngram.get_data(data)
    ngram.compute_score()

    # nchar = nchar(3)
    # nchar.get_data(data)
    # nchar.compute_score()

    # Cos = cosine()
    # Cos.get_data(data)
    # Cos.compute_score()

    # pdb.set_trace()
