'''
Created on Sep 13, 2015


@author: ananta
'''
from sklearn.tree import DecisionTreeClassifier
import logging
from mashable_data import getMashableData, getMashableMatrix, SimpleTimer
from sklearn import tree
from sklearn.learning_curve import learning_curve
from plot_learning_curve import plot_learning_curve
from sklearn.metrics import confusion_matrix
logger = logging.getLogger('training')

def printPdf(clf, dataTrain):
    from sklearn.externals.six import StringIO
    import pydot
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('sentiment.pdf')
    print dataTrain.data[0]

def runDecisionTreeSimulation(dataTrain, dataTest, train_M, test_M):
    print 'running decision tree'
    outFile = open('decisionTreeLog.txt','a')
    
    
    outFile.write('train==> %d, %d \n'%(train_M.shape[0],train_M.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_M.shape[0],test_M.shape[1]))
    with SimpleTimer('time to train', outFile):
        clf = DecisionTreeClassifier().fit(train_M, dataTrain.target)
    
    baseScore = clf.score(test_M, dataTest.target)
    initHeight = clf.tree_.max_depth
    
    print 'baseline score %.3f base height %d' % (baseScore, initHeight)
    outFile.write('baseline score %.3f base height %d \n' % (baseScore, initHeight))
    
    res = []
    with SimpleTimer('time to prune', outFile):
        for height in range(initHeight, 2 , -1):
#             print 'training for height %d' % height
            clf = DecisionTreeClassifier(max_depth=height).fit(train_M, dataTrain.target)
            score = clf.score(test_M, dataTest.target)
            res.append((score, height))
            outFile.write('%d %.3f \n' % (height, score))
    res = sorted(res, key=lambda x:x[0], reverse=True)
    print res[:5]
    '''
    train_sizes, train_scores, valid_scores = learning_curve(DecisionTreeClassifier(), train_M, dataTrain.target, train_sizes=[50, 80, 110], cv=5)
    print train_sizes
    print train_scores
    print valid_scores
    '''
    bestDepth = res[0][1]
    print ('best height is %d' % bestDepth)
    outFile.write('best depth is %d  and score is %d \n' % (bestDepth, res[0][0]))
    bestClf = DecisionTreeClassifier(max_depth=bestDepth)
    bestClf.fit(train_M, dataTrain.target)
    predicted = bestClf.predict(test_M)
    trainPredict = bestClf.predict(train_M)
    print len(filter(lambda x:x==0, dataTrain.target)), len(filter(lambda x:x==0, trainPredict))
    print len(filter(lambda x:x==1, dataTrain.target)), len(filter(lambda x:x==1, trainPredict))
    print len(filter(lambda x:x==2, dataTrain.target)), len(filter(lambda x:x==2, trainPredict))
    print confusion_matrix(trainPredict, dataTrain.target)
    print confusion_matrix(predicted, dataTest.target)
    
    results = predicted == dataTest.target
    wrong = []
    for i in range(len(results)):
        if not results[i]:
            wrong.append(i)
    print 'classifier got these wrong:'
    for i in wrong[:10]:
        print dataTest.data[i][0], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i][0], dataTest.target[i]))
    plot_learning_curve(bestClf, 'decision tree after pruning from %d to %d depth' % (initHeight, bestDepth), train_M, dataTrain.target, cv=5, n_jobs=4)
    
if __name__ == '__main__':
    dataSize = 50000
    dataTrain, dataTest = getMashableData(dataSize)
    train_M, test_M = getMashableMatrix(dataTrain, dataTest)
    runDecisionTreeSimulation(dataTrain, dataTest, train_M, test_M)