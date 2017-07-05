from __future__ import division
import operator
from functools import reduce
from nb_trainedData import TrainedData
from nb_helper import Helper

class NBClassifier:

    def __init__(self):
        '''

        :param helper:
        :param default_prob:
        :return:
        '''
        self.data = TrainedData()
        self.helper = None
        self.defaultProb = 1e-9

    def fit(self, feature_train, label_train):
        '''

        :param corpus:
        :return:
        '''
        #corpus = self.get_corpus_cat_text_format(features_train, labels_train)
        precalculate = 0
        self.helper = Helper()


        txt = feature_train
        className = label_train
        self.data.increaseClass(className)
        tokens = self.helper.tokenize(txt)
        for token in tokens:
            self.data.increaseToken(token, className)

        # update featrue_sample_count
        self.data.update_feature_sample_count(tokens, className)

    def predict(self, text):
        '''

        :param text:
        :return:
        '''
        tokens = self.helper.tokenize(text)
        classes = self.data.getClasses()
        probsOfClasses = {}
        i = 0
        for className in classes:
            # we are calculating the probablity of seeing each token
            # in the text of this class
            # P(Token_1|Class_i)
            tokensProbs = [self.getTokenProb(token, className) for token in tokens]
            # calculating the probablity of seeing the the set of tokens
            # in the text of this class
            # P(Token_1|Class_i) * P(Token_2|Class_i) * ... * P(Token_n|Class_i)
            try:
                tokenSetProb = reduce(lambda a, b: a * b, (i for i in tokensProbs if i))
            except:
                tokenSetProb = 0
            probsOfClasses[className] = [tokenSetProb * self.getPrior(className)]

        cls_probs = sorted(probsOfClasses.items(), key=operator.itemgetter(1), reverse=True)
        label, probability = cls_probs[0]
        probability = probability[0]
        return label, probability

    def getPrior(self, className):
        '''

        :param className:
        :return:
        '''
        return self.data.getClassDocCount(className) /  self.data.getDocCount()


    def getTokenProb(self, token, className):
        '''

        :param token:
        :param className:
        :return:
        '''
        if token == 'stop_word':
            return None
        # p(token|Class_i)
        classDocumentCount = self.data.getClassDocCount(className)

        # if the token is not seen in the training set, so not indexed,
        # then we return None not to include it into calculations.
        try:
            tokenFrequency = self.data.getFrequency(token, className)
        except:
            return None

        # print("Before if tokenFrequency is None")
        # this means the token is not seen in this class but others.
        if tokenFrequency is None:
            return self.defaultProb

        probablity = tokenFrequency / classDocumentCount
        return probablity