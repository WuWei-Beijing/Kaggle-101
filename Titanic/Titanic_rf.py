import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
path="C:\\Users\\wei\\Desktop\\Kaggle\\Kaggle101\\Titanic\\"  
  
############################step1:import the data#################################################    
def loadDataFrame():
    train_df=pd.read_csv(path+'train.csv')
    test_df=pd.read_csv(path+'test.csv')
    df=pd.concat([train_df,test_df])
    df.reset_index(inplace=True)
    df.drop('index',axis=1,inplace=True)
    #merge the train DataFrame and the test DataFrame because we 
    #need more data to do statical things
    #reindex the columns to be the columns in the training data
    df=df.reindex_axis(train_df.columns,axis=1)
    print df.shape[1],"columns:",df.columns.values
    print "Row count:",df.shape[0]
    return df

#########################step2:generating the features###########################################
###Generate feature from the 'Plclass' variable
def processPclass(df,keep_binary=False,keep_scaled=False):
    #fill in the missing value
    df['Pclass'][df.Pclass.isnull()]=df['Pclass'].median()
    #create binary features
    if keep_binary:
        df=pd.concat([df,pd.get_dummies(df['Pclass']).rename(columns=lambda x:'Pclass_'+str(x))],axis=1)
    if keep_scaled:
        scaler=preprocessing.StandardScaler()
        df['Pclass_scaled']=scaler.fit_transform(df['Pclass'])
    del df['Pclass']
    return df

###Generate features from the 'Name' variable
def processName(df,keep_binary=False,keep_scaled=False,keep_bins=False):
    """
    Parameters:
        keep_binary:include 'Title_Mr' 'Title_Mrs'...
        keey_scaled&&keep_bins:include 'Names_scaled' 'Title_id_scaled'
    Note: the string feature 'Name' can be deleted
    """
    # how many different names do they have? this feature 'Names'
    df['Names']=df['Name'].map(lambda x:len(re.split('\\(',x)))
    
    #what is each person's title? 
    df['Title']=df['Name'].map(lambda x:re.compile(", (.*?)\.").findall(x)[0])
    #group low-occuring,related titles together
    df['Title'][df.Title == 'Jonkheer'] = 'Master'
    df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
    df['Title'][df.Title == 'Mme'] = 'Mrs'
    df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
    df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
    #build binary features
    if keep_binary:
        df=pd.concat([df,pd.get_dummies(df['Title']).rename(columns=lambda x:'Title_'+str(x))],axis=1)
    #process_scaled
    if keep_scaled:
        scaler=preprocessing.StandardScaler()
        df['Names_scaled']=scaler.fit_transform(df['Names'])
        del df['Names']
    if keep_bins:
        df['Title_id']=pd.factorize(df['Title'])[0]+1
        del df['Title']
    if keep_bins and keep_scaled:
        scaler=preprocessing.StandardScaler()
        df['Title_id_scaled']=scaler.fit_transform(df['Title_id'])
        del df['Title_id']
    del df['Name']
    return df

###Generate feature from 'Sex' variable
def processSex(df):
    df['Gender'] = np.where(df['Sex'] == 'male', 1, 0)
    del df['Sex']
    return df

###Generate feature from 'SibSp' and 'Parch'
def processFamily(df,keep_binary=False,keep_scaled=False):
    #interaction variables require no zeros ,lift up everything
    df['SibSp']=df['SibSp']+1
    df['Parch']=df['Parch']+1
    if keep_binary:
        sibsps=pd.get_dummies(df['SibSp']).rename(columns=lambda x:'SibSp_'+str(x))
        parchs=pd.get_dummies(df['Parch']).rename(columns=lambda x:'Parch_'+str(x))
        df=pd.concat([df,sibsps,parchs],axis=1)
    if keep_scaled:
        scaler=preprocessing.StandardScaler()
        df['SibSp_scaled']=scaler.fit_transform(df['SibSp'])
        df['Parch_scaled']=scaler.fit_transform(df['Parch'])
    del df['SibSp']
    del df['Parch']
    return df
    
###Generate features from 'Ticket' variable
###Utility method: get the index of 'Ticket'
def getTicketPrefix(ticket):
    match=re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:
        return match.group(0)
    else:
        return 'U'

###Utility method: get the numerical component of 'Ticket'
def getTicketNumber(ticket):
    match=re.compile("([0-9]+)").search(ticket)
    if match:
        return match.group(0)
    else:
        return '0'
###Generate features of 'Ticket'
def processTicket(df,keep_binary=False,keep_bins=False,keep_scaled=False):
    df['TicketPrefix']=df['Ticket'].map(lambda x:getTicketPrefix(x.upper()))
    df['TicketPrefix']=df['TicketPrefix'].map(lambda x:re.sub('[\.?\/?]','',x))
    df['TicketPrefix']=df['TicketPrefix'].map(lambda x:re.sub('STON','SOTON',x))
    
    df['TicketNumber']=df['Ticket'].map(lambda x:getTicketNumber(x))
    df['TicketNumberLen']=df['TicketNumber'].map(lambda x:len(x)).astype(np.int)
    df['TicketNumberStart']=df['TicketNumber'].map(lambda x:x[0]).astype(np.int)
    
    if keep_binary:
        prefixes = pd.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
        numberlen = pd.get_dummies(df['TicketNumberLen']).rename(columns=lambda x: 'TicketNumberLen_' + str(x))
        numberstart = pd.get_dummies(df['TicketNumberStart']).rename(columns=lambda x: 'TicketNumberStart_' + str(x))
        df = pd.concat([df, prefixes,numberlen,numberstart], axis=1)
    if keep_bins:
        #help the interactive feature process,lift by 1
        df['TicketPrefix_id']=pd.factorize(df['TicketPrefix'])[0]+1      
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['TicketPrefix_id_scaled'] = scaler.fit_transform(df['TicketPrefix_id'])
        df['TicketNumberLen_scaled'] = scaler.fit_transform(df['TicketNumberLen'])
        df['TicketNumberStart_scaled'] = scaler.fit_transform(df['TicketNumberStart'])
    del df['Ticket'],df['TicketNumber'],df['TicketPrefix'],df['TicketPrefix_id'],df['TicketNumberLen'],df['TicketNumberStart']
    return df

def setMissingAges(df):
    age_df=df[['Age','Embarked','Fare','Parch','SibSp','Title_id','Pclass','Names','CabinLetter','Sex']]
    knownAge=age_df[df.Age.notnull()]
    unknownAge=age_df[df.Age.isnull()]
    y=knownAge.values[:,0]
    X=knownAge.values[:,1:]
    rfr=RandomForestRegressor(n_estimators=2000,n_jobs=-1)
    #train the regressor
    rfr.fit(X,y)
    predictedAges=rfr.predict(unknownAge[:,1:])
    df['Age'][df.Age.isnull()]=predictedAges
    return df

def fillDataFrame(df):
    #fill with zero
    df['Cabin'][df.Cabin.isnull()]='u0'
    #fill with average value
    df['Fare'][df.Fare.isnull()]=df['Fare'].median()
    #fill with the most common value
    df['Embarked'][df.Embarked.isnull()]=df['Embarked'].dropna().value_counts().argmax()
    #fill with predicting througth regression
    #df=setMissingAges(df)
    df=processPclass(df,keep_scaled=True)
    df=processName(df,keep_bins=True,keep_scaled=True)
    df=processSex(df)
    df=processFamily(df,keep_scaled=True)
    df=processTicket(df,keep_bins=True,keep_scaled=True)
    print df
    return df

df=loadDataFrame()
df=fillDataFrame(df)