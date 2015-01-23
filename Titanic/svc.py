import loaddata
import numpy as np
import csv
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
from operator import itemgetter
path="C:\\Users\\wei\\Desktop\\Kaggle\\Kaggle101\\Titanic\\" 

def report(grid_scores,n_top=10):
    """
    Output a simple report of the top parameter sets from hyperparameter optimization
    """
    params=None
    #grid_scores is a list of named tuples. Each named tuple has the attributes:
    #-----------parameters, a dict of parameter settings
    #-----------mean_validation_score, the mean score over the cross-validation folds
    #-----------cv_validation_scores, the list of scores for each fold
    top_scores=sorted(grid_scores,key=itemgetter(1),reverse=True)[:n_top]
    for i,score in enumerate(top_scores):
        print("Parameters with rank:{0}".format(i+1))
        print("Mean validation score:{0:.4f}(std:{1:.4f})".format(score.mean_validation_score,np.std(score.cv_validation_scores)))
        print("Parameters:{0}".format(score.parameters))
        if params==None:
            params=score.parameters# record the highest score
    return params

def Titanic_svc():
    print '\nUsing Support Vector Machine, Generating initial training/test sets'
    train_df,test_df=loaddata.getData(keep_binary=True,keep_bins=True,keep_scaled=False)
    #print '\n',train_df.columns.size,'Feed in features of SVM',train_df.columns.values
    #save the 'PassengerId' column
    test_ids=test_df['PassengerId']
    train_df.drop('PassengerId',axis=1,inplace=1)
    test_df.drop('PassengerId',axis=1,inplace=1)
    features_list=train_df.columns.values[1:]
    X=train_df.values[:,1:]
    y=train_df.values[:,0]
    X_test=test_df.values
    svc=SVC()
    ########################Step5: Reduce initial feature set with estimated feature importance####
    print "\nfeed SVM with all binary features"
    ########################Step6:Parameter tunning with CrossValidation(RandomSearch)###########
    ###Random search the best parameter
    """
    rbf_params = {"kernel": ['rbf'],
                    "class_weight": ['auto'],
                    "C": [0.5,1,3,5],
                    "gamma": [0.01,0.05,0.1,0.5],
                    "tol": 10.0**-np.arange(2,4),
                    "random_state": [1234567890]}
    
    poly_params = {"kernel": ['poly'],
                    "class_weight": ['auto'],
                    "degree": [1,3,5,7],
                    "C": [0.5,1,3,5],
                    "gamma": 10.0**np.arange(-1, 1),
                    "coef0": 10.0**-np.arange(1,2),
                    "tol": 10.0**-np.arange(1,3),
                    "random_state": [1234567890]}
   
    sigmoid_params = {"kernel": ['sigmoid'],
                        "class_weight": ['auto'],
                        "C": 10.0**np.arange(-2,6),
                        "gamma": 10.0**np.arange(-3, 3),
                        "coef0": 10.0**-np.arange(1,5),
                        "tol": 10.0**-np.arange(2,4),
                        "random_state": [1234567890]}
    print "Hyperparameter opimization using RandomizedSearchCV..."
    rand_search=RandomizedSearchCV(svc,param_distributions=poly_params,n_jobs=7,cv=3,n_iter=1000)
    rand_search.fit(X,y)
    best_params=report(rand_search.grid_scores_)
    params=best_params
    """
    #==========================The best tunned parameters=========================================
    
    params_score = {"kernel": 'rbf',
                    "class_weight": 'auto',
                    "C": 3,
                    "gamma": 0.1,
                    "tol": 0.01,
                    "random_state": 1234567890}
    """
    params_score = {"kernel": 'poly',
                    "class_weight": 'auto',
                    "degree":3,
                    "C": 3,
                    "gamma": 0.1,
                    "coef0": 0.1,
                    "tol": 0.01,
                    "random_state": 1234567890}
    """
    params=params_score
    
    #============================================================================================
    
    ########################Step7:Model generation/validation(Learning curve/Roc curve)#############
    print "Generating RandomForestClassifier model with parameters:",params
    svc=SVC(**params)
    ###Predict the accuracy on test set(hold some data of training set to test)
    print "\nCalculating the Accuracy..."
    test_accs=[]
    for i in range(5):
        X_train,X_hold,y_train,y_hold=train_test_split(X,y,test_size=0.3)
        svc.fit(X_train,y_train)
        acc=svc.score(X_hold,y_hold)
        print "\nAccuracy is:{:.4f}".format(acc)
        test_accs.append(acc)
    acc_mean="%.3f"%(np.mean(test_accs))
    acc_std="%.3f"%(np.std(test_accs))
    print "\nmean accuracy:",acc_mean,"and stddev:",acc_std
    ########################Step8:Predicting and Saving result######################################
    svc.fit(X,y)
    return test_ids,svc.predict(X_test),float(acc_mean)
    
if __name__=='__main__':
    test_ids,result,acc_mean=Titanic_svc()
    submission=np.asarray(zip(test_ids,result)).astype(int)
    #ensure passenger IDs in ascending order
    output=submission[submission[:,0].argsort()]
    predict_file=open(path+"predict.csv",'wb')
    file_object=csv.writer(predict_file)
    file_object.writerow(["PassengerId","Survived"])
    file_object.writerows(output)
    predict_file.close()
    print 'Done'