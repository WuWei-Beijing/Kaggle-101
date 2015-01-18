import csv
import numpy as np
path="C:\\Users\\wei\\Desktop\\Kaggle\\Kaggle101\\Digit Recognizer\\"
def readCSVFile(file):
    rawData=[]
    trainFile=open(path+file,'rb')
    reader=csv.reader(trainFile)
    for line in reader:
        rawData.append(line)#42001 lines,the first line is header
    rawData.pop(0)#remove header
    intData=np.array(rawData).astype(np.int32)
    return intData
    
def loadTrainingData():
    intData=readCSVFile("train.csv")
    label=intData[:,0]
    data=intData[:,1:]
    data=np.where(data>0,1,0)#replace positive in feature vector to 1
    return data,label

def loadTestData():
    intData=readCSVFile("test.csv")
    data=np.where(intData>0,1,0)
    return data

def loadTestResult():  
    intData=readCSVFile("knn_benchmark.csv")
    data=np.mat(intData)
    return data[:,1]
    
#inX marks the point we want to classify
#labels marks the class each known point is within
#k marks the number we want to pick to be most closest to our test point
def classify(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=np.array(diffMat)**2
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
        myFile=open(path+"result.csv",'wb')    
        myWriter=csv.writer(myFile)
        myWriter.writerow(['ImageId','Label'])
        ind=range(len(result))
        for i,val in zip(ind,result):
            line=[]
            line.append(i+1)
            line.append(val)
            myWriter.writerow(line)
            
def handwritingClassTest():
    trainData,trainLabel=loadTrainingData()
    testData=loadTestData()
    testLabel=loadTestResult()
    m,n=np.shape(testData)
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