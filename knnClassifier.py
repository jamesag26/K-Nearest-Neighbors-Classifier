# By James Alford-Golojuch
import random, math
from scipy.spatial.distance import cdist
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

# define our hypothesis
def f(x): 
  return expit(np.matrix([0, 1, -.5,.5])*x);

def getkfolds(k,dataList):
    # Places all of training data in one list to be randomly ditributed into k sets
    # Randomly shuffles data
    random.shuffle(dataList)
    # Finds size of K equal sets
    lenPart=int(math.ceil(len(dataList)/float(k)))
    # Divides dataset into k partitions of size lenPart
    partTrain = {}
    for x in range(k):
        partTrain[x] = dataList[x*lenPart:x*lenPart+lenPart]
    return partTrain

def knn(k, testPoint, points, labels):
    kNearestNeighbors = []
    # Counter
    x = 0
    for instance in points:
        temp = [[0], [0]]
        trainPoint = instance
        #trainPoint = [(instance[0], instance[1])]
        y = cdist(testPoint, trainPoint, 'euclidean')
        temp[0]=y[0][0]
        temp[1] = labels[x]
        #Adds each distance from test point and classification for each training point 
        kNearestNeighbors.append(temp[:])
        x = x + 1
    # Sorts KNN to find k nearest neighbots
    kNearestNeighbors.sort(key=itemgetter(0))
    label0Count = 0
    label1Count = 0
    # Tallies up labels for nearest neighbors
    for num in range(0,k):
        if kNearestNeighbors[num][1] == 0:
            label0Count = label0Count + 1
        else:
            label1Count = label1Count + 1
    # Making sure correct number of neighbors are being pulled
    '''print ("total k neighbors polled")
    print (label1Count+label0Count)'''
    # Returns value for label with highest tally or randomly picks if tied
    if label0Count > label1Count:
        return 0
    elif label0Count < label1Count:
        return 1
    else:
        return random.randint(0,1)
        
def averageErrorRates(file):
    classifierData = []
    
    # Assign values for each instance to list
    #houseData = open('data/housing_classification.data', 'r')
    houseData = open(file, 'r')
    kFolds = 10
    for lines in houseData:
        lines = lines.strip()
        vars = lines.split(',')
        tempList = []
        for x in vars:
            tempList.append(float(x))
        classifierData.append(tempList)
    # Partition data into 10 kfolds
    partData = getkfolds(kFolds,classifierData)
    
    # Total error rates for each k neighbor
    # Each position in list will contain a list of 10 error rates for each kfold
    errorRates = [[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], 
                  [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                  [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1], [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                  [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]]
    
    
    # Will find misclassification for each kNN (1-10) for each kfold
    for num in range(0,10):
        # Divides into testing and training data for each kfold
        testingData = partData[num]
        trainingData = [x for x in classifierData if x not in testingData]
        # testing to show testing data != training data
        '''print("Testing Data")
        print(testingData)
        print("Training Data")
        print(trainingData)'''
        # Lists for labels and features
        trainingLabels = []
        trainingPoints = []
        testingLabels = []
        testingPoints = []
        x = []
        y = []
        # Seperates labels and features
        for item in trainingData:
            x.append(item[0])
            y.append(item[1])
            temp = [(item[0], item[1])]
            trainingLabels.append(item[2])
            trainingPoints.append(temp)
        for item in testingData:
            temp = [(item[0], item[1])]
            testingLabels.append(item[2])
            testingPoints.append(temp)
        #print (trainingLabels)
        #print (trainingPoints)
        # Gets misclassification rate for each k nearest neighbor (k=1-10)
        for k in range (1,11):
            # Gets prediction for each data point and compares to actual label
            # Total predictions
            total = 0
            # Total misclassifications
            misclass = 0
            for dataPoint in testingPoints:
                prediction = knn(k, dataPoint, trainingPoints, trainingLabels)
                if prediction != int(testingLabels[total]):
                    misclass = misclass + 1
                total = total + 1
            errorRates[k-1][num] = (misclass/total)
            x = np.array(x)
            y = np.array(y)
            if file == 'data/synthetic-1.csv' and k == 1:
                plt.title("Synthetic 1: k = 1")
                #x_min, x_max = x.min() - 1, x.max() + 1
                #y_min, y_max = y.min() - 1, y.max() + 1
                #x_min = -15
                #x_max = 15
                #y_min = -15
                #y_max = 15
                #xx, yy = np.meshgrid(np.arange(x_min, x_max),
                #     np.arange(y_min, y_max))
                #Z = f(np.c_[xx.ravel(), yy.ravel()])
                #Z = Z.reshape(xx.shape)
                #plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
                #plt.axis('off')  
                
                # Plot also the training points
                #plt.scatter(x, y, cmap=plt.cm.Paired)
                
                plt.scatter(x, y, c=trainingLabels)
                plt.show()
            if file == 'data/synthetic-1.csv' and k == 3:
                plt.title("Synthetic 1: k = 3")
                plt.scatter(x, y, c=trainingLabels)
                plt.show()
            if file == 'data/synthetic-2.csv' and k == 1:
                plt.title("Synthetic 2: k = 1")
                plt.scatter(x, y, c=trainingLabels)
                plt.show()
            if file == 'data/synthetic-2.csv' and k == 3:
                plt.title("Synthetic 2: k = 3")
                plt.scatter(x, y, c=trainingLabels)
                plt.show()
            if file == 'data/synthetic-3.csv' and k == 1:
                plt.title("Synthetic 3: k = 1")
                plt.scatter(x, y, c=trainingLabels)
                plt.show()
            if file == 'data/synthetic-3.csv' and k == 3:
                plt.title("Synthetic 3: k = 3")
                plt.scatter(x, y, c=trainingLabels)
                plt.show()
            if file == 'data/synthetic-4.csv' and k == 1:
                plt.title("Synthetic 4: k = 1")
                plt.scatter(x, y, c=trainingLabels)
                plt.show()
            if file == 'data/synthetic-4.csv' and k == 3:
                plt.title("Synthetic 4: k = 3")
                plt.scatter(x, y, c=trainingLabels)
                plt.show()
    # Finds Average error rate for each k neighbor
    errorRatesAvg = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    for i in range(0,10):
        temp = 0
        for kFolds in errorRates[i]:
            temp = temp + kFolds
        errorRatesAvg[i] = temp/10
    print (errorRatesAvg)
    return
        
#print (errorRates)
print("Synthetic Dataset 1")
averageErrorRates('data/synthetic-1.csv')
print("Synthetic Dataset 2")
averageErrorRates('data/synthetic-2.csv')
print("Synthetic Dataset 3")
averageErrorRates('data/synthetic-3.csv')
print("Synthetic Dataset 4")
averageErrorRates('data/synthetic-4.csv')
