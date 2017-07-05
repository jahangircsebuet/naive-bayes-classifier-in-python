from nbClassifier import NBClassifier
import time
from pprint import pprint

############################Custom Naive Bayes Classifier##########################
print("Custom Naive Bayes part")
cout = 0
corpus = []
tuple_list = []
classifier = NBClassifier()
time_list = []
time_fit_start = time.localtime()
time_list.append(time_fit_start)
with open('training-data-large.txt', 'r') as infile:
    for line in infile:
        cat = line[0]
        txt = line[2:].strip()
        classifier.fit(txt, cat)
        tuple_list.append({'text': txt, 'category': cat})

    infile.close()

time_fit_end = time.localtime()
time_list.append(time_fit_end)

corpus = []
label_y = []
count = 0
isTrueCount = 0
new_corpus = []
wrong_pred = []
time_pred_start = time.localtime()
time_list.append(time_pred_start)
with open('training-data-large.txt', 'r') as infile:
    for line in infile:
        corpus.append(line[2:].strip())
        label_y.append(line[0])
        count +=1
        act = line[0]
        pred = classifier.predict(line[2:].strip())

        pred = pred[0]
        if act == pred:
            isTrueCount +=1
            isTrue = True
        else:
            wrong_pred.append((act, line[2:].strip()))
            isTrue = False
        #print('Actual: ', act, 'predicted: ', pred, isTrue)
    infile.close()

    print('\n')
    print('Total Sample: ', count)
    print('Correct Prediction: ', isTrueCount)
    print(float(isTrueCount*100)/count)
    print("wrong_pred: ", len(wrong_pred))

    cor_pred = 0
    iteration = 0
    while float(isTrueCount * 100) / count < 95:
        print("\nInside while loop...")
        count = 0
        isTrueCount = 0

        for item in wrong_pred:
            corpus.append(item[1])
            label_y.append(item[0])
            #print("inside for: len(corpus): ", len(corpus))
        i = 0
        for item in corpus:
            classifier.fit(item, label_y[i])
            i +=1

        i = 0
        print("Len(corpus): ", len(corpus))
        print("Len(label_y): ", len(label_y))

        wrong_pred = []
        for item in corpus:
            count += 1
            pred = classifier.predict(item)
            pred = pred[0]
            if act == pred:
                isTrueCount += 1
                isTrue = True
            else:
                wrong_pred.append((act, item))
                isTrue = False

        print('\n')
        print('Total Sample: ', count)
        print('Correct Prediction: ', isTrueCount)
        print("% ", float(isTrueCount * 100) / count)
        print("wrong_pred: ", len(wrong_pred))
        iteration +=1

    print("Iteration: ", iteration)
    time_pred_end = time.localtime()
    time_list.append(time_pred_end)

    pprint(time_list)
######################################NLTK Naive Bayes Classifier##############################
print('\n')
print("NLTK part")
import nltk

def feature_extractor_contains(sentence):
    '''
    returns the features exists or not
    :param sentence:
    :return:
    '''
    features = {}
    for word in sentence.split(','):
        features["contains({})".format(word)] = 1
    return features

def feature_extractor_count(sentence):
    '''
    returns the count of features
    :param sentence:
    :return:
    '''
    features = {}
    feature_list = []
    for word in sentence.split(','):
        if word not in feature_list:
            feature_list.append(word)
            features["count({})".format(word)] = 1
        else:
            features["count({})".format(word)] += 1

    return features

featuresets = []
for tuples in tuple_list:
    txt = tuples['text']
    cat = tuples['category']
    featuresets.append((feature_extractor_contains(txt), cat))

pprint(len(featuresets))

classifier = nltk.NaiveBayesClassifier.train(featuresets)
print(nltk.classify.accuracy(classifier, featuresets))

