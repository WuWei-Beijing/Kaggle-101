import csv
from numpy import *
def normalize(data):
    m,n=shape(data)
    for i in range(m):
        for j in range(n):
            if data[i,j]!=0:
                data[i,j]=1
    return data
    
def loadTrainingData():
    rawData=[]
    trainFile=open("C:\\Users\\wei\\Desktop\\Kaggle\\Kaggle101\\Digit Recognizer\\train.csv",'rb')
    reader=csv.reader(trainFile)
    for line in reader:
        rawData.append(line)#42001 lines,the first line is header
    rawData.pop(0)#remove header
    intData=[]
    for line in rawData:
        intLine=[]
        for stringNumber in line:
            intLine.append(int(stringNumber))#string-->int
        intData.append(intLine)
    intMat=mat(intData)
    label=intMat[:,0]
    data=normalize(intMat[:,1:])
    return data[:10,:],label[:10]
    #return data,label

def loadTestData():
    rawData=[]
    testFile=open("C:\\Users\\wei\\Desktop\\Kaggle\\Kaggle101\\Digit Recognizer\\test.csv",'rb')
    reader=csv.reader(testFile)
    for line in reader:
        rawData.append(line)
    rawData.pop(0)#remove header
    intData=[]
    for line in rawData:
        intLine=[]
        for stringNumber in line:
            intLine.append(int(stringNumber))#string-->int
        intData.append(intLine)
    data=normalize(mat(intData))
    #return data
    return data[:10]

def loadTestResult():
    rawData=[]
    resultFile=open("C:\\Users\\wei\\Desktop\\Kaggle\\Kaggle101\\Digit Recognizer\\knn_benchmark.csv",'rb')
    reader=csv.reader(resultFile)
    for line in reader:
        rawData.append(line)
    rawData.pop(0)#remove header
    intData=[]
    for line in rawData:
        intLine=[]
        for stringNumber in line:
            intLine.append(int(stringNumber))#string-->int
        intData.append(intLine)
    data=mat(intData)
    return data[:,1]
    
#inX marks the point we want to classify
#labels marks the class each known point is within
#k marks the number we want to pick to be most closest to our test point
def classify(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=array(diffMat)**2
    sqDistance=sqDiffMat.sum(axis=1)
    distance=sqDistance**0.5
    sortedDistanceIndices=distance.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistanceIndices[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=lambda jj:jj[1],reverse=True)
    return sortedClassCount[0][0]

def saveResult(result):
        myFile=open("C:\\Users\\wei\\Desktop\\Kaggle\\Kaggle101\\Digit Recognizer\\result.csv",'wb')    
        myWriter=csv.writer(myFile)
        for i in result:
            line=[]
            line.append(i)
            myWriter.writerow(line)
            
def handwritingClassTest():
    trainData,trainLabel=loadTrainingData()
    testData=loadTestData()
    testLabel=loadTestResult()
    m,n=shape(testData)
    errorCount=0
    resultList=[]
    for i in range(m):
         classifierResult = classify(testData[i], trainData, trainLabel, 5)
         resultList.append(classifierResult)
         print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[i])
         if (classifierResult != testLabel[i]): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(m))
    saveResult(resultList)
    
handwritingClassTest()