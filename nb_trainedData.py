import sys
from nb_exception_not_seen import NotSeen
from pprint import pprint

class TrainedData(object):
    def __init__(self):
        '''

        '''
        self.docCountOfClasses = {}
        self.frequencies = {}
        self.stop_words = []


    def increaseClass(self, className):
        '''

        :param className:
        :param byAmount:
        :return:
        '''
        self.docCountOfClasses[className] = self.docCountOfClasses.get(className, 0) + 1


    def increaseToken(self, token, className):
        '''

        :param token:
        :param className:
        :param byAmount:
        :return:
        '''
        if not token in self.frequencies:
                self.frequencies[token] = {}

        if not className in self.frequencies[token].keys():
            self.frequencies[token][className] =  [0, 0]
            self.frequencies[token][className][0] += 1
        else:
            self.frequencies[token][className][0] +=1

    def increaseBigramToken(self, token, className):
        '''

        :param token:
        :param className:
        :param byAmount:
        :return:
        '''
        if not token in self.frequencies:
                self.frequencies[token] = {}
        if not className in self.frequencies[token].keys():
            self.frequencies[token][className] =  [0, 0]
            self.frequencies[token][className][0] += 1
        else:
            self.frequencies[token][className][0] +=1

    def update_feature_sample_count(self, tokens, className):
        '''

        :param tokens: per sample tokens
        :param className:
        :return:
        '''
        counted_tokens = []
        for token in tokens:
            if not token in counted_tokens:
                counted_tokens.append(token)
                self.frequencies[token][className][1] +=1


    def get_feature_sample_count(self, tokens, className):
        '''

        :param tokens:
        :param className:
        :return:
        '''
        sum = 0
        input_feature_count = 0
        for token in tokens:
            input_feature_count +=1
            sum += self.frequencies[token][className][0] * self.frequencies[token][className][1]
        return sum - input_feature_count


    def decreaseToken(self, token, className, byAmount=1):
        '''

        :param token:
        :param className:
        :param byAmount:
        :return:
        '''
        if token not in self.frequencies:
            raise NotSeen(token)
        foundToken = self.frequencies[token]
        if className not in self.frequencies:
            sys.stderr.write("Warning: token %s has no entry for class %s. Not decreasing.\n" % (token, className))
            return
        if foundToken[className] < byAmount:
            raise ArithmeticError("Could not decrease %s/%s count (%i) by %i, "
                                  "as that would result in a negative number." % (
                                      token, className, foundToken[className], byAmount))
        foundToken[className] -= byAmount

    def getDocCount(self):
        '''
        returns all documents count
        :return:
        '''
        return sum(self.docCountOfClasses.values())

    def getClasses(self):
        '''
        returns the names of the available classes as list
        :return:
        '''
        return self.docCountOfClasses.keys()

    def getClassDocCount(self, className):
        '''
        returns document count of the class.
        If class is not available, it returns None
        :param className:
        :return:
        '''
        return self.docCountOfClasses.get(className, None)

    def getFrequency(self, token, className):
        '''
        returns token count for a given class
        :param token:
        :param className:
        :return:
        '''
        if token in self.frequencies:
            foundToken = self.frequencies[token]

            if foundToken.get(className) == None:
                return None
            else:
                return foundToken.get(className)[0]
        else:
            raise NotSeen(token)
