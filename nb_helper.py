import sys
import re
import random
from nltk import stem
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import wordpunct_tokenize

lemmatizer = WordNetLemmatizer()

class Helper(object):
    def __init__(self, corpus=[], precalculate=0):
        '''

        :param corpus:
        '''
        self.signs_to_remove = ["?!#%&."]
        self.stop_words = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()

        self.sample_count = {}
        self.feature_count = {}
        self.class_count = 0
        self.total_sample_count = 0
        self.featureset_per_class = {}

        self.all_pair_classes_common_features = {}
        self.all_pair_classes_uncommon_features = {}

        self.common_featureset_distinct_length = {}
        self.uncommon_featureset_distinct_length = {}

        self.common_featureset_count = 0
        self.uncommon_featureset_count = 0

#        self.precalculate(corpus, precalculate)


    def precalculate(self, corpus, precalculate):
        '''

        :param corpus:
        :param precalculate:
        :return:
        '''
        # count per class sample #
        if precalculate == 1:
            cls_count = 0
            for tuple in corpus:
                self.total_sample_count += 1
                if not tuple['category'] in self.sample_count:
                    self.sample_count[tuple['category']] = 1
                    cls_count += 1
                else:
                    self.sample_count[tuple['category']] += 1
            self.class_count = cls_count

            # count per class feature #
            # calculate self.feature_count[token][cat] values
            for tuple in corpus:
                text = tuple['text']
                cat = tuple['category']
                if cat not in self.featureset_per_class:
                    self.featureset_per_class[cat] = []
                tokens = self.preprocess_text(self.tokenize(text))

                for token in tokens:
                    if self.featureset_per_class[cat].count(token) == 0:
                        self.featureset_per_class[cat].append(token)
                    if not token in self.feature_count:
                        self.feature_count[token] = {}
                        if cat not in self.feature_count[token]:
                            self.feature_count[token][cat] = 1
                        else:
                            self.feature_count[token][cat] += 1
                    else:
                        if cat not in self.feature_count[token]:
                            self.feature_count[token][cat] = 1
                        else:
                            self.feature_count[token][cat] += 1

            # calculate uncommon featuresets between all pair classes and store the feature sets
            uncommon_featureset_count = 0
            for key_1 in self.featureset_per_class.keys():
                self.all_pair_classes_common_features[key_1] = {}
                self.all_pair_classes_uncommon_features[key_1] = {}
                for key_2 in self.featureset_per_class.keys():
                    if key_1 != key_2:

                        s_1 = set(self.featureset_per_class[key_1])
                        s_2 = set(self.featureset_per_class[key_2])

                        common_features = s_1.intersection(s_2)
                        uncommon_features_1 = s_1 - common_features
                        uncommon_features_2 = s_2 - common_features

                        if len(common_features) != 0:
                            self.all_pair_classes_common_features[key_1][key_2] = []
                            self.common_featureset_count += 1
                            if len(common_features) not in self.common_featureset_distinct_length:
                                self.common_featureset_distinct_length[len(common_features)] = 1
                            else:
                                self.common_featureset_distinct_length[len(common_features)] += 1
                            self.all_pair_classes_common_features[key_1][key_2].append(list(common_features))

                        if len(uncommon_features_1) != 0 and len(uncommon_features_2) != 0:
                            self.all_pair_classes_uncommon_features[key_1][key_2] = []
                            self.all_pair_classes_uncommon_features[key_1][key_2].append(list(uncommon_features_1))
                            self.all_pair_classes_uncommon_features[key_1][key_2].append(list(uncommon_features_2))
                            self.uncommon_featureset_count += 1


    def remove_stop_words(self,token):
        '''

        :param token:
        :return:
        '''
        if token in self.stop_words:
            return "stop_word"
        else:
            return token


    def remove_non_alpha(self, token):
        '''

        :param token:
        :return:
        '''
        regex = re.compile('[^a-zA-Z]')
        str = regex.sub('', token)
        if len(str) == 0:
            return "stop_word"
        else:
            return str


    def remove_punctuation(self, token):
        '''

        :param token:
        :return:
        '''
        if len(re.sub(str(self.signs_to_remove), "", token)) == 0:
            return "stop_word"
        else:
            return re.sub(str(self.signs_to_remove), "", token)


    def tokenize(self, text):
        '''

        :param text:
        :return:
        '''
        return text.lower().split(',')


    def preprocess_text(self, tokens):
        '''

        :param tokens:
        :return:
        '''
        processed_tokens = []
        for token in tokens:
            token = self.remove_punctuation(token)
            if token != 'stop_word':
                token = self.remove_non_alpha(token)
            if token != 'stop_word':
                token = self.remove_stop_words(token)
            if token != 'stop_word':
                token = self.get_lemmatized_with_pos_tagging(token)
            if token != 'stop_word':
                processed_tokens.append(token)
        return processed_tokens


    def get_lemmatized_with_pos_tagging(self, token):
        '''

        :param token:
        :return:
        '''
        tokenized = wordpunct_tokenize(token)
        tok, tag = pos_tag(tokenized)[0]
        tag = {
            "N": wn.NOUN,
            "V": wn.VERB,
            "R": wn.ADV,
            "J": wn.ADJ
        }.get(tag[0], wn.NOUN)

        lemmatized = self.lemmatizer.lemmatize(tok, tag)
        return lemmatized


    def get_stemmed_token(self, token):
        '''

        :param token:
        :return:
        '''
        #stemmer = PorterStemmer()
        #return stemmer.stem(token)
        stemmer = stem.lancaster.LancasterStemmer()
        return stemmer.stem(token)