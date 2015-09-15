'''
Created on Sep 14, 2015

@author: ananta
'''
import numpy
from matplotlib import pyplot
import numpy

if __name__ == '__main__':
    res = []
    fileName = 'OnlineNewsPopularity.csv'
    with open(fileName,'r') as inFile:
            inpts = inFile.readline()
            j = 2
            while True:
                inpts = inFile.readline()
                if inpts.strip() == '':
                    break
                line = inpts.split(',')[:-2] # remove the popularity score before feeding it in
                tar = float(line[-1])
                if tar < 0.5:
                    res.append(tar) # remove the title url
    
    print len(res)
    
    y = numpy.random.rand(len(res))
    x = numpy.array(res)
    pyplot.scatter(x,y)
    
    pyplot.show()
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.mlab as mlab
    import math
    
    mean = 3395.380184
    variance = 135182573.7
    sigma = math.sqrt(variance)
    x = np.linspace(1,843300,500)
    plt.plot(x,mlab.normpdf(x,mean,sigma))
    
    plt.show()
    '''
