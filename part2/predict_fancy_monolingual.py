
import sys
import pdb
import os
import nltk
from features import Data, ngram, nchar, cosine, alignment
import numpy as np
from sklearn.linear_model import Ridge

class Similarity(object):
    def __init__(self, feature_pick, input_file):
        self.input_file = input_file
        self.feature_pick = feature_pick
        self.data = Data(input_file)
        self.feature_list = ['align', 'cos', 'ngram', 'nchar', 'cardinality']
        self.feature_map = {'align':alignment(), 'cos':cosine(), 'ngram': ngram(2), 'nchar': nchar(3), 'cardinality':None}
        self.sim_score = np.array([])

    def compute_sim_score(self):
        featureObj = [self.feature_map[self.feature_list[i]] for i in self.feature_pick]
        score = []
        for obj in featureObj:
            obj.get_data(self.data)
            score.append(obj.compute_score())

        self.sim_score = np.array(score).T
        return self.sim_score

    def write_score_tofile(self,output_file = 'output.txt', with_golden = False):
        print 'writing scores to file ...'
        score_data = self.sim_score
        if with_golden:
            self.golden_score = self.data.read_score().reshape(-1,1)
            score_data = np.concatenate((score_data, self.golden_score), axis=1)
        try:
            os.remove(output_file)
        except OSError:
            pass
        np.savetxt(output_file, score_data)

class Regression(object):
    def __init__(self, feature_pick = [0,1]):
        self.feature_pick = feature_pick
        self.alpha = 205
        self.X = None
        self.Y = None

    def load_train_data(self, dataName_list):
        for name in dataName_list:
            train_data = np.loadtxt(name)
            data = train_data[:,self.feature_pick]
            golden_score = train_data[:,-1]
            if self.X is None:
                self.X = data
                self.Y = golden_score
            else:
                self.X = np.concatenate((self.X, data), axis = 0)
                self.Y = np.concatenate((self.Y, golden_score), axis = 0)


    def ridge_regress(self):
        self.clf = Ridge(alpha = self.alpha)
        self.clf.fit(self.X,self.Y)

    def predict_score(self, fileName):
        self.ridge_regress()
        data = np.loadtxt(fileName)
        test_data = data[:,self.feature_pick]
        s = self.clf.predict(test_data)
        s[s<0] = 0
        s[s>5] = 5
        self.score = s

    def write_score_tofile(self, output_file):
        print 'writing predict score to file ...'
        try:
            os.remove(output_file)
        except OSError:
            pass
        np.savetxt(output_file, self.score.reshape(-1,1))



if __name__ == "__main__":
    args = sys.argv
    print args
    Sim = Similarity([0,1,2,3], args[1])
    sim_scores = Sim.compute_sim_score()
    name = ''.join( ['test', args[1].split('.')[-2], '.txt'] )
    Sim.write_score_tofile(name)

    reg = Regression([0,1,2,3])
    prefix = 'stopwordsdata/'
    train_data = [prefix+x for x in ['stopwordTrainMSRpar.txt', 'stopwordTrainMSRvid.txt', 'stopwordTrainMSRvid.txt'] ]
    reg.load_train_data(train_data)
    reg.predict_score(name)
    reg.write_score_tofile(args[-1])
