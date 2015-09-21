'''
Created on Sep 13, 2015

@author: ananta
'''

'''
Created on Sep 12, 2015

@author: ananta
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mashable_data import getMashableData, getMashableMatrix, SimpleTimer, getPickeledData, outputScores
from plot_learning_curve import plot_learning_curve

def tryVariousHyperParams(dataTrain, dataTest, train_M, test_M, outFile):
    params = [(8,200),(8,400),(8,600),(8,800),(8,1000),(8,1200),(8,1600),(8,2000)]
    res = []
    outFile.write('\n changing num_estimators \n')
    for depth, num in params:
        outFile.write('depth %d, num %d \n'%(depth, num))
        with SimpleTimer('time to train', outFile):
            estimator = DecisionTreeClassifier(max_depth=depth)
            clf = AdaBoostClassifier(base_estimator=estimator,  n_estimators=num)
            clf.fit(train_M, dataTrain.target)
            
        score = clf.score(test_M, dataTest.target)
        print depth, num
        print score
        outFile.write('score %.3f \n'%(score))
        res.append((score, depth, num))
    res = sorted(res, key=lambda x:x[0])
    return res[0]

def runBoosting(dataTrain, dataTest, holdout, train_M, test_M, hold_M):
    outFile = open('boostingLog.txt','a')
    print 'running boosting algo'
    outFile.write('train==> %d, %d \n'%(train_M.shape[0],train_M.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_M.shape[0],test_M.shape[1]))
    # takes a very long time to run
#     score, bestDepth, num = tryVariousHyperParams(dataTrain, dataTest, train_M, test_M)
    bestDepth = 7
    bestNum = 10000
    with SimpleTimer('time to train', outFile):
        estimator = DecisionTreeClassifier(max_depth=bestDepth)
        bestClf = AdaBoostClassifier(base_estimator=estimator,  n_estimators=bestNum)
        bestClf.fit(train_M, dataTrain.target)
    
    bestScore = bestClf.score(test_M, dataTest.target)
    print 'the best score %.3f' % bestScore
    outFile.write('depth %d, num %d score %.3f \n'%(bestDepth, bestNum, bestScore))
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
    plot_learning_curve(bestClf, 'boosting with %d trees' % bestNum, train_M, dataTrain.target, cv=3, n_jobs=4)
    
if __name__ == '__main__':
#     dataSize = 30000
    train, test, holdout, train_M, test_M, hold_M = getPickeledData(fileName='sample.p')
    runBoosting(train, test, holdout, train_M, test_M, hold_M)
    #outFile = open('boostingLogHyper.txt','a')
    #tryVariousHyperParams(train, test,train_M, test_M,outFile)
    
