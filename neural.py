'''
Created on Sep 13, 2015

@author: ananta
'''

from mashable_data import getMashableData, getMashableMatrix, SimpleTimer
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules   import SoftmaxLayer, TanhLayer, SigmoidLayer, LinearLayer
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities           import percentError
from pybrain.structure import RecurrentNetwork, FullConnection, FeedForwardNetwork

def getDataSetFromTfidf(tfidf, target):
    ds = ClassificationDataSet(tfidf.shape[1], nb_classes=3, class_labels=['Unpopular','Neutral','Popular'])
    for i in range(tfidf.shape[0]):
        ds.addSample(tfidf[i], [target[i]])
    ds._convertToOneOfMany()
    return ds
        
def runNeuralSimulation(dataTrain, dataTest, train_M, test_M):
    outFile = open('neuralLog.txt','a')
    outFile.write('-------------------------------------\n')
    outFile.write('train==> %d, %d \n'%(train_M.shape[0],train_M.shape[1]))
    outFile.write('test==>  %d, %d \n'%(test_M.shape[0],test_M.shape[1]))
    
    trainDS = getDataSetFromTfidf(train_M, dataTrain.target)
    testDS = getDataSetFromTfidf(test_M, dataTest.target)
    
    print "Number of training patterns: ", len(trainDS)
    print "Input and output dimensions: ", trainDS.indim, trainDS.outdim
    print "First sample (input, target, class):"
    print len(trainDS['input'][0]), trainDS['target'][0], trainDS['class'][0]
    print 'input layer', trainDS.indim
    '''
    with SimpleTimer('time to train', outFile):
        net = buildNetwork(54, 18, 3, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer)
        trainer = BackpropTrainer( net, dataset=trainDS, momentum=0.01, verbose=True, weightdecay=0.01, batchlearning=True)
    
    '''
    net = FeedForwardNetwork()
    net.addInputModule(LinearLayer(54, name='in'))
    net.addModule(SigmoidLayer(18, name='hidden'))
    net.addOutputModule(SoftmaxLayer(3, name='out'))
    net.addConnection(FullConnection(net['in'], net['hidden'], name='c1'))
    net.addConnection(FullConnection(net['hidden'], net['out'], name='c2'))
#     net.addRecurrentConnection(FullConnection(net['hidden'], net['hidden'], name='c3'))
    net.sortModules()
    
    trainer = BackpropTrainer( net, dataset=trainDS, momentum=0.01, verbose=True, weightdecay=0.01)
    
    outFile.write('%s \n' % (net.__str__()))
    epochs = 2000
    with SimpleTimer('time to train %d epochs' % epochs, outFile):
        for i in range(epochs):
            trainer.trainEpochs(1)
            trnresult = percentError( trainer.testOnClassData(),
                                  trainDS['class'] )
            tstresult = percentError( trainer.testOnClassData(
               dataset=testDS ), testDS['class'] )
    
            print "epoch: %4d" % trainer.totalepochs, \
                  "  train error: %5.2f%%" % trnresult, \
                  "  test error: %5.2f%%" % tstresult
    outFile.write('%5.2f , %5.2f \n' % (100.0-trnresult, 100.0-tstresult))
                  
    predicted = trainer.testOnClassData(dataset=testDS)
    results = predicted == testDS['class'].flatten()
    wrong = []
    for i in range(len(results)):
        if not results[i]:
            wrong.append(i)
    print 'classifier got these wrong:'
    for i in wrong[:10]:
        print dataTest.data[i][0], dataTest.target[i]
        outFile.write('%s %d \n' % (dataTest.data[i], dataTest.target[i]))
    
if __name__ == '__main__':
    dataSize = 10000
    dataTrain, dataTest = getMashableData(dataSize)
    train_M, test_M = getMashableMatrix(dataTrain, dataTest, 54)
    runNeuralSimulation(dataTrain, dataTest, train_M, test_M)
    