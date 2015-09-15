'''
Created on Sep 13, 2015

@author: ananta
'''

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from plot_learning_curve import plot_learning_curve
from mashable_data import getMashableData, getMashableMatrix, SimpleTimer

def runSVMSimulation(dataTrain, dataTest, train_M, test_M):
    outFile = open('svmLog.txt','a')
    print 'running svm code'
    outFile.write('train==> %d, %d \n'%(train_M.shape[0],train_M.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_M.shape[0],test_M.shape[1]))
    kernel = "linear"
    with SimpleTimer('time to train', outFile):
#         clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=30, random_state=42)
#         clf = LinearSVC(C=1.0)
        clf = SVC(kernel=kernel, C=0.025)
        clf.fit(train_M, dataTrain.target)
    
    baseScore = clf.score(test_M, dataTest.target)
    baseIter = 5
    print 'baseline score %.3f base iter %d' % (baseScore, baseIter)
    outFile.write('baseline score %.3f base iter %d \n' % (baseScore, baseIter))
    
    res = []
    with SimpleTimer('number of iter', outFile):
        for itr in range(1):
            print 'training for neighbors %d' % itr
            clf = SVC(kernel=kernel, C=0.025)
#             clf = LinearSVC(loss='squared_hinge', C=1.0)
            clf.fit(train_M, dataTrain.target)
            score = clf.score(test_M, dataTest.target)
            res.append((score, itr))
            outFile.write('%d %.3f \n' % (itr, score))
    res = sorted(res, key=lambda x:x[0], reverse=True)
    print res[:5]
    bestItr = res[0][1]
    print ('best number of iter is %d' % bestItr) 
    bestClf = SVC(kernel=kernel, C=0.025)
    bestClf.fit(train_M, dataTrain.target)
    predicted = bestClf.predict(test_M)
    
    results = predicted == dataTest.target
    res = []
    for i in range(len(results)):
        if not results[i]:
            res.append(i)
    print 'classifier got these wrong:'
    for i in res[:10]:
        print dataTest.data[i], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i], dataTest.target[i]))
        

    bestClf = SVC(kernel=kernel, C=0.025)
    plot_learning_curve(bestClf, 'svm with %s kernel' % kernel, train_M, dataTrain.target, cv=5, n_jobs=4)
    '''
    bestClf = LinearSVC(loss='hinge', C=1.0)
    plot_learning_curve(bestClf, 'svm hinge with %d iter' % bestItr, train_M, dataTrain.target, cv=5, n_jobs=4)
    '''
        
if __name__ == '__main__':
    dataSize = 1000
    dataTrain, dataTest = getMashableData(dataSize)
    train_M, test_M = getMashableMatrix(dataTrain, dataTest)
    runSVMSimulation(dataTrain, dataTest, train_M, test_M)