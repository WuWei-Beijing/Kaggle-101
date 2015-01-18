import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    intData=readCSVFile("rf_benchmark.csv")
    data=np.mat(intData)
    return data[:,1]
    
def saveResult(result):
        myFile=open(path+"result.csv",'wb')    
        myWriter=csv.writer(myFile)
        myWriter.writerow(['ImageId','Label'])
        ind=range(len(result))
        for i,val in zip(ind,result):
            line=[]
            line.append(i+1)
            for v in val:
                line.append(v)
            myWriter.writerow(line)
            
def handwritingClassTest():
    #load data and normalization
    trainData,trainLabel=loadTrainingData()
    testData=loadTestData()
    testLabel=loadTestResult()
    #train the rf classifier
    clf=RandomForestClassifier(n_estimators=1000,min_samples_split=5)
    clf=clf.fit(trainData,trainLabel)#train 20 objects
    m,n=np.shape(testData)
    errorCount=0
    resultList=[]
    for i in range(m):#test 5 objects
         classifierResult = clf.predict(testData[i])
         resultList.append(classifierResult)
         print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[i])
         if (classifierResult != testLabel[i]): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(m))
    saveResult(resultList)
   
handwritingClassTest()