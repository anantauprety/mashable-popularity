'''
Created on Sep 13, 2015

@author: ananta
'''

from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC, LinearSVC
from plot_learning_curve import plot_learning_curve
from mashable_data import getMashableData, getMashableMatrix, SimpleTimer, getPickeledData, outputScores

def runSVMSimulation(dataTrain, dataTest, holdout, train_M, test_M, hold_M):
    kernel = "linear"
    outFile = open('svmSarinLog%s.txt' % kernel,'a')
    print 'running svm code'
    
    outFile.write('train==> %d, %d \n'%(train_M.shape[0],train_M.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_M.shape[0],test_M.shape[1]))
    
    penalty = 0.025
    with SimpleTimer('time to train', outFile):
#         clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=30, random_state=42)
#         clf = LinearSVC(C=1.0)
        clf = SVC(kernel=kernel, C=penalty, degree=1)
        clf.fit(train_M, dataTrain.target)
    
    baseScore = clf.score(test_M, dataTest.target)
    baseIter = 5
    print 'baseline score %.3f base iter %d' % (baseScore, baseIter)
    outFile.write('baseline score %.3f base iter %d \n' % (baseScore, baseIter))
    
    res = []
    with SimpleTimer('number of iter', outFile):
        for pen in [1,5,10,15,20,30]:
            print 'training for neighbors %.3f' % pen
            clf = SVC(kernel=kernel, C=pen, degree=1)
#             clf = LinearSVC(loss='squared_hinge', C=1.0)
            clf.fit(train_M, dataTrain.target)
            score = clf.score(hold_M, holdout.target)
            res.append((score, pen))
            trainPredict = clf.score(train_M, dataTrain.target)
            outFile.write('test %.3f %.3f \n' % (pen, score))
            outFile.write('train %.3f %.3f \n' % (pen, trainPredict))
            
    res = sorted(res, key=lambda x:x[0], reverse=True)
    print res[:5]
    
    bestPen = res[0][1]
    print ('best number of iter is %.3f' % bestPen) 
    
    bestClf = SVC(kernel=kernel, C=penalty, degree=bestPen)
    bestClf.fit(train_M, dataTrain.target)
    
    predicted = bestClf.predict(test_M)
    
    trainPredict = bestClf.predict(train_M)
    
    print 'testing score'
    outFile.write('testing score')
    outputScores(dataTest.target, predicted, outFile)
    
    print 'training score'
    outFile.write('training score')
    outputScores(dataTrain.target, trainPredict, outFile)
    
    results = predicted == dataTest.target
    res = []
    for i in range(len(results)):
        if not results[i]:
            res.append(i)
    print 'classifier got these wrong:'
    for i in res[:10]:
        print dataTest.data[i], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i], dataTest.target[i]))
        

    
    plot_learning_curve(bestClf, 'svm with %s kernel & penalty %.3f' % (kernel, bestPen), train_M, dataTrain.target, cv=5, n_jobs=4)
    '''
    bestClf = LinearSVC(loss='hinge', C=1.0)
    plot_learning_curve(bestClf, 'svm hinge with %d degree' % bestItr, train_M, dataTrain.target, cv=5, n_jobs=4)
    '''
        
if __name__ == '__main__':
    dataSize = 30000
    train, test, holdout, train_M, test_M, hold_M = getPickeledData(fileName='sample.p')
    runSVMSimulation(train, test, holdout, train_M, test_M, hold_M)
