import loaddata
import numpy as np
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import RandomizedSearchCV
import matplotlib.pyplot as plt
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

def Titanic_adbst():
    print '\nUsing Adaptive Boosting Descision Tree, Generating initial training/test sets'
    train_df,test_df=loaddata.getData(keep_binary=True,keep_bins=True)
    #print '\n',train_df.columns.size,'Feed in features of Adaboost DT',train_df.columns.values
    #save the 'PassengerId' column
    test_ids=test_df['PassengerId']
    train_df.drop('PassengerId',axis=1,inplace=1)
    test_df.drop('PassengerId',axis=1,inplace=1)
    features_list=train_df.columns.values[1:]
    X=train_df.values[:,1:]
    y=train_df.values[:,0]
    X_test=test_df.values
    ########################Step5: Reduce initial feature set with estimated feature importance####
    print "\nRough fitting a Adaboost DT to determine feature importance"
    clf=AdaBoostClassifier(n_estimators=1000,learning_rate=0.01,random_state=1234567890).fit(X,y)
    feature_importance=clf.feature_importances_
    feature_importance=100.0*(feature_importance/feature_importance.max())
    fi_threshold=5
    important_idx=np.where(feature_importance>fi_threshold)[0]
    important_features=features_list[important_idx]
    #print "\n", important_features.shape[0], "Important features(>", fi_threshold, "percent of max importance)...\n",important_features
    sorted_idx=np.argsort(feature_importance[important_idx])[::-1]
    #plot feature importance
    """
    pos=np.arange(sorted_idx.shape[0])+0.5
    plt.subplot(1,2,2)
    plt.barh(pos,feature_importance[important_idx][sorted_idx[::-1]],align='center')
    plt.yticks(pos,important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.draw()
    plt.show()
    """
    #Remove non-important features from the feature set
    X=X[:,important_idx][:,sorted_idx]
    X_test=X_test[:,important_idx][:,sorted_idx]
    #print "\nSorted (DESC) Useful X:\n",X
    test_df=test_df.iloc[:,important_idx].iloc[:,sorted_idx]
    print '\nTraining with', X.shape[1], "features:\n", test_df.columns.values
    ########################Step6:Parameter tunning with CrossValidation(RandomSearch)###########
    ###Random search the best parameter
    #==========================The best tunned parameters=========================================
    
    params_score={"n_estimators":1000,"learning_rate":0.1,"random_state": 1234567890}
    params=params_score
    
    #============================================================================================
    
    ########################Step7:Model generation/validation(Learning curve/Roc curve)#############
    print "Generating AdaBoost model with parameters:",params
    clf=AdaBoostClassifier(**params)
    ###Predict the accuracy on test set(hold some data of training set to test)
    print "\nCalculating the Accuracy..."
    test_accs=[]
    for i in range(5):
        X_train,X_hold,y_train,y_hold=train_test_split(X,y,test_size=0.3)
        clf.fit(X_train,y_train)
        acc=clf.score(X_hold,y_hold)
        print "\nAccuracy is:{:.4f}".format(acc)
        test_accs.append(acc)
    acc_mean="%.3f"%(np.mean(test_accs))
    acc_std="%.3f"%(np.std(test_accs))
    print "\nmean accuracy:",acc_mean,"and stddev:",acc_std
    ########################Step8:Predicting and Saving result######################################
    clf.fit(X,y)
    return test_ids,clf.predict(X_test),float(acc_mean)
    
if __name__=='__main__':
    test_ids,result,acc_mean=Titanic_adbst()
    submission=np.asarray(zip(test_ids,result)).astype(int)
    #ensure passenger IDs in ascending order
    output=submission[submission[:,0].argsort()]
    predict_file=open(path+"predict.csv",'wb')
    file_object=csv.writer(predict_file)
    file_object.writerow(["PassengerId","Survived"])
    file_object.writerows(output)
    predict_file.close()
    print 'Done'