'''
Created on Sep 13, 2015

@author: ananta
'''

from sklearn.neighbors import KNeighborsClassifier
from mashable_data import getMashableData, getMashableMatrix, SimpleTimer
from plot_learning_curve import plot_learning_curve
import numpy

def runKNNSimulation(dataTrain, dataTest, train_M, test_M):
    outFile = open('knnLog.txt','a')
    print 'running mashable knn simulation'
    outFile.write('train==> %d, %d \n'%(train_M.shape[0],train_M.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_M.shape[0],test_M.shape[1]))
    with SimpleTimer('time to train', outFile):
        clf = KNeighborsClassifier(weights='distance', ).fit(train_M, dataTrain.target)
    
    baseScore = clf.score(test_M, dataTest.target)
    baseParams = clf.get_params(True)
    baseNeighbors = baseParams['n_neighbors']
    print 'baseline score %.3f base n_neighbors %d' % (baseScore, baseNeighbors)
    outFile.write('baseline score %.3f base height %d \n' % (baseScore, baseNeighbors))
    
    res = []
    with SimpleTimer('time to fine tune number of neighbors', outFile):
        for neighbors in range(2,baseNeighbors * 10):
#             print 'training for neighbors %d' % neighbors
            clf = KNeighborsClassifier(n_neighbors=neighbors, weights='distance').fit(train_M, dataTrain.target)
            score = clf.score(test_M, dataTest.target)
            res.append((score, neighbors))
            outFile.write('%d %.3f \n' % (neighbors, score))
    res = sorted(res, key=lambda x:x[0], reverse=True)
    print res[:5]
    bestNeighbors = res[0][1]
    print ('best number of neighbors is %d' % bestNeighbors)
    outFile.write('best number of neighbors is %d  and score is %d\n' % (bestNeighbors, res[0][0]))
    bestClf = KNeighborsClassifier(n_neighbors=bestNeighbors, weights='distance')
    bestClf.fit(train_M, dataTrain.target)
    predicted = bestClf.predict(test_M)
    
    results = predicted == dataTest.target
    print numpy.mean(results)
    res = []
    for i in range(len(results)):
        if not results[i]:
            res.append(i)
    print 'classifier got these wrong:'
    for i in res[:10]:
        print dataTest.data[i], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i], dataTest.target[i]))
    '''
    train_sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(), train_M, dataTrain.target, train_sizes=[50, 80, 110], cv=5)
    print train_sizes
    print train_scores
    print valid_scores
    '''
       
    plot_learning_curve(bestClf, 'knn with %d neighbors' % bestNeighbors, train_M, dataTrain.target, cv=5, n_jobs=4)
    
if __name__ == '__main__':
    dataSize = 30000
    dataTrain, dataTest = getMashableData(dataSize)
    train_M, test_M = getMashableMatrix(dataTrain, dataTest)
    runKNNSimulation(dataTrain, dataTest, train_M, test_M)
