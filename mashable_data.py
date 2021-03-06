'''
Created on Sep 13, 2015

@author: ananta
'''
from random import shuffle, choice
from sklearn.datasets.base import Bunch
import time
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import os.path
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
import pickle


FEATURE_NAMES = [' timedelta', ' n_tokens_title', ' n_tokens_content', 
                 ' n_unique_tokens', ' n_non_stop_words', 
                 'n_non_stop_unique_tokens', ' num_hrefs', 
                 ' num_self_hrefs', ' num_imgs', ' num_videos', 
                 ' average_token_length', ' num_keywords', ' data_channel_is_lifestyle', 
                 ' data_channel_is_entertainment', ' data_channel_is_bus', 
                 ' data_channel_is_socmed', ' data_channel_is_tech', ' data_channel_is_world', 
                 'kw_min_min', ' kw_max_min', ' kw_avg_min', ' kw_min_max', ' kw_max_max', 
                 ' kw_avg_max', ' kw_min_avg', ' kw_max_avg', ' kw_avg_avg', 
                 ' self_reference_min_shares', ' self_reference_max_shares', 
                 ' self_reference_avg_sharess', ' weekday_is_monday', ' weekday_is_tuesday', 
                 ' weekday_is_wednesday', ' weekday_is_thursday', ' weekday_is_friday', 
                 ' weekday_is_saturday', ' weekday_is_sunday', ' is_weekend', ' LDA_00', 
                 ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04', ' global_subjectivity', 
                 ' global_sentiment_polarity', ' global_rate_positive_words', 
                 ' global_rate_negative_words', ' rate_positive_words', ' rate_negative_words', 
                 ' avg_positive_polarity', ' min_positive_polarity', ' max_positive_polarity', 
                 ' avg_negative_polarity', ' min_negative_polarity', ' max_negative_polarity', 
                 ' title_subjectivity', ' title_sentiment_polarity', ' abs_title_subjectivity', 
                 ' abs_title_sentiment_polarity']

class SimpleTimer(object):

    def __init__(self, txt, outFile=None):
        self.text = txt or ''
        self.outFile = outFile

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        if self.outFile:
            self.outFile.write('%s %.3f \n' % (self.text, time.time() - self.start))
        print self.text, time.time() - self.start

def outputScores(y_true, y_predicted, outFile):
    
    test_conf = confusion_matrix(y_true, y_predicted)
    
    print test_conf
    
    print 'accuracy score %.3f' % accuracy_score(y_true, y_predicted)
    print 'f1 score %.3f' % f1_score(y_true, y_predicted)
    print 'recall score %.3f' % recall_score(y_true, y_predicted)
    
    outFile.write('confusion matrix test: \n %s \n' % (test_conf))
    
    outFile.write('accuracy score %.3f \n' % accuracy_score(y_true, y_predicted))
    outFile.write('f1 score %.3f \n' % f1_score(y_true, y_predicted))
    outFile.write('recall score %.3f \n' % recall_score(y_true, y_predicted))

class MashableData(object):
    
    def randomList(self, a):
        b = []
        for i in range(len(a)):
            element = choice(a)
            a.remove(element)
            b.append(element)
        return b
    
    def __init__(self, fileName='mashable.csv'):
        self.names = ['unpopular','neutral','popular']
        self.dataDict = {}
        self.train, self.test, self.holdout = self.getData(fileName)
        self.feature_names = None
    
    def dummyCoding(self, num):
#         firstQuartile = -0.211
#         secondQuartile = -0.05
        firstQuartile = -0.225
        secondQuartile = 0.1
        if num == 'unpopular':
            return 0 # neutral
        elif num == 'popular':
            return 1 # unpopular
       
        
    def getData(self, fileName):
        res = []
        neutralCount = 0
        with open(fileName,'r') as inFile:
            inpts = inFile.readline()
            print inpts
            j = 2
            while True:
                inpts = inFile.readline()
                if inpts.strip() == '':
                    break
                line = inpts.split(',') # remove the popularity score before feeding it in
                if len(line) != 59:
                    print j                
                self.dataDict[j] = line[0]
                j += 1
                tar = line[-1]
                tar = self.dummyCoding(tar.strip())
                
                res.append((tar,line[:-2])) # remove the title url
                
            shuffle(res)
            
        print 'total data size %d' % len(res)
        # we want 60% training, 20% testing and 20% holdout set
        first = int(0.6 * len(res))
        third = len(res) - int(0.2 * len(res))
        train = res[:first]
        x = res[first:]
        x = self.randomList(x)
        test, holdout = x[:len(x)/2], x[len(x)/2:]
        return train, test, holdout
    
    def fetchData(self, subset='train', n_sample=10):
        if subset == 'train':
            return self.shuffleData(self.train[:n_sample])
        elif subset == 'test':
            return self.shuffleData(self.test[:n_sample])
        elif subset == 'holdout':
            return self.shuffleData(self.holdout[:n_sample])
            
    def shuffleData(self, res):
        shuffle(res)
        train = Bunch()
        train.data = map(lambda x:x[1], res)
        train.target = map(lambda x:x[0], res)
        train.target_names = self.names
        return train
        

def getMashableData(size=10, ratio=0.2):
#     filePath = 'C:\\Users\\ananta\\Documents\\6220\\Sentiment-Analysis-Dataset\\TwitterSentiment.txt'
    mashData = MashableData()
    
    with SimpleTimer('time to fetch training data'):
        dataTrain = mashData.fetchData(subset='train', n_sample=int(size-size*ratio))
        print '%.3f unpopular on training data' % (len(filter(lambda x:x == 0, dataTrain.target)) * 1.0 / len(dataTrain.target) * 100)
        print '%.3f neutral on training data' % (len(filter(lambda x:x == 1, dataTrain.target)) * 1.0 / len(dataTrain.target) * 100)
        print '%.3f popular on training data' % (len(filter(lambda x:x == 2, dataTrain.target)) * 1.0 / len(dataTrain.target) * 100)
    with SimpleTimer('time to fetch testing data'):
        dataTest = mashData.fetchData(subset='test', n_sample=int(size*ratio))
    with SimpleTimer('time to fetch holdout data'):
        dataHoldOut = mashData.fetchData(subset='holdout', n_sample=int(size*ratio))
    return dataTrain, dataTest, dataHoldOut

def getMashableMatrix(dataTrain, dataTest, dataHold, chooseK='all'):
    
    print 'calculating training matrix'
    X_train = []
    for data in dataTrain.data:
        X_train.append(map(lambda x:float(x), data[1:]))
        
    
    print 'calculating test matrix'
    X_test = []
    for data in dataTest.data:
        X_test.append(map(lambda x:float(x), data[1:]))
    
    print 'calculating holdout matrix'
    X_hold = []
    for data in dataHold.data:
        X_hold.append(map(lambda x:float(x), data[1:]))
        
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, dataTrain.target)
    X_test = scaler.transform(X_test)
    X_hold = scaler.transform(X_hold)
    
#     normalizer = Normalizer()
#     X_train = normalizer.fit_transform(X_train, dataTrain.target)
#     X_test = normalizer.transform(X_test)
    print 'train', len(X_train), len(X_train[0])
    print 'test', len(X_test), len(X_test[0])
    print 'hold', len(X_hold), len(X_hold[0])
    
    kBest = SelectKBest(f_classif, k=chooseK)
    
    X_train = kBest.fit_transform(X_train, dataTrain.target)
    X_test = kBest.transform(X_test)
    X_hold = kBest.transform(X_hold)
    if FEATURE_NAMES:
        # keep selected feature names
        print len(FEATURE_NAMES)
        feature_names = [FEATURE_NAMES[i] for i
                         in kBest.get_support(indices=True)]
    print feature_names[:10]
    print 'train after feature selection', len(X_train), len(X_train[0])
    print 'test after feature selection', len(X_test), len(X_test[0])
    print 'hold after feature selection', len(X_hold), len(X_hold[0])
    return X_train, X_test, X_hold

def getPickeledData(fileName='sample.p'):
    allData = pickle.load(open(fileName, 'rb'))
    return allData['dataTrain'], allData['dataTest'], allData['dataHold'], allData['train_M'],  allData['test_M'], allData['hold_M']
    
def pickleData(dataTrain, dataTest, holdOut, trainTfidf, testTfidf, holdOutTfidf, fileName='sample.p'):
    allData = {'dataTrain':dataTrain,'dataTest':dataTest,'train_M':trainTfidf,'test_M':testTfidf, 'dataHold':holdOut, 'hold_M':holdOutTfidf}
    pickle.dump(allData, open(fileName, 'wb'))
    

def printStats(target):
    print '%.3f unpopular on training data' % (len(filter(lambda x:x == 0, target)))
    print '%.3f neutral on training data' % (len(filter(lambda x:x == 1, target)))
    print '%.3f popular on training data' % (len(filter(lambda x:x == 2, target)))
    
if __name__ == '__main__':
    train, test, holdout = getMashableData(30000)
    print len(train.data), len(test.data), len(holdout.data)
    train_M, test_M, hold_M = getMashableMatrix(train, test, holdout, chooseK='all')
    print len(train_M), len(test_M), len(hold_M)
    pickleData(train, test, holdout, train_M, test_M, hold_M, fileName='sample.p')
    train, test, holdout, train_M, test_M, hold_M = getPickeledData(fileName='sample.p')
    print len(train.target), len(test.target), len(holdout.target), len(train_M)
    print 'training'
    printStats(train.target)
    print 'holdout'
    printStats(holdout.target)
    print 'testing'
    printStats(test.target)
    