'''
Created on Sep 13, 2015

@author: ananta
'''

from sklearn.neighbors import KNeighborsClassifier
from mashable_data import getMashableData, getMashableMatrix, SimpleTimer, getPickeledData, outputScores
from plot_learning_curve import plot_learning_curve
import numpy

def runKNNSimulation(dataTrain, dataTest, holdout, train_M, test_M, hold_M):
    outFile = open('knnLog25.txt','a')
    print 'running mashable knn simulation'
    outFile.write('train==> %d, %d \n'%(train_M.shape[0],train_M.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_M.shape[0],test_M.shape[1]))
    with SimpleTimer('time to train', outFile):
        clf = KNeighborsClassifier(weights='distance', ).fit(train_M, dataTrain.target)
    plot_learning_curve(clf, 'knn with %d neighbors' , train_M, dataTrain.target, cv=5, n_jobs=4)
    
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
            score = clf.score(hold_M, holdout.target)
            res.append((score, neighbors))
            outFile.write('%d %.3f \n' % (neighbors, score))
    res = sorted(res, key=lambda x:x[0], reverse=True)
    print res[:5]
    bestNeighbors = res[0][1]
    print ('best number of neighbors is %d' % bestNeighbors)
    outFile.write('best number of neighbors is %d  and score is %.3f\n' % (bestNeighbors, res[0][0]))
    
    bestClf = KNeighborsClassifier(n_neighbors=bestNeighbors, weights='distance')
    bestClf.fit(train_M, dataTrain.target)
    
    predicted = bestClf.predict(test_M)
    trainPredict = bestClf.predict(train_M)
    print 'testing score'
    outFile.write('testing score')
    outputScores(dataTest.target, predicted, outFile)
    print 'training score'
    outFile.write('testing score')
    outputScores(dataTrain.target, trainPredict, outFile)
    
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
    train, test, holdout, train_M, test_M, hold_M = getPickeledData(fileName='sample.p')
    runKNNSimulation(train, test, holdout, train_M, test_M, hold_M)
    '''
    import pandas
    from numpy import dtype,array
    df = pandas.read_csv('dress.csv')
    # print df.info()
    # print df.describe()
    
    dt =  df.dtypes
    newCols = {}
    for t in dt.iteritems():
         if t[1] == dtype('object'):
             newCols[t[0]] = pandas.factorize(df[t[0]])[0]
         else:
             newCols[t[0]] = df[t[0]].values
    # print objs
    dfN = pandas.DataFrame(newCols)
    dfN = dfN.drop(['Recommendation','Dress_ID','Size'],axis=1)
    val = dfN.values
    print dfN.columns
    target = df['Recommendation']
    target_names = target.unique().astype(str)
    target = target.values.astype(int)
    from sklearn.datasets.base import Bunch
    from sklearn.decomposition import PCA,KernelPCA,RandomizedPCA,FastICA
    # clf = PCA(n_components=10)
    # clf = RandomizedPCA(n_components=5,whiten=True)
    clf = FastICA(n_components=10,whiten=True)
    # kernel = "linear" | "poly" | "rbf" | "sigmoid" | "cosine" 
    # clf = KernelPCA(n_components=2,kernel='sigmoid')
    print clf
    val = clf.fit_transform(val)
    iris = Bunch()
    iris.data = val
    iris.target = target
    iris.target_names = target_names
    em(iris)
    '''
