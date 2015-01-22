import loaddata
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import csv
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

if __name__=='__main__':
    print "\nGenerating initial training/test sets"
    train_df,test_df=loaddata.getData(keep_binary=True,keep_bins=True,keep_scaled=True,keep_interactive=True)
    #save the 'PassengerId' column
    test_ids=test_df['PassengerId']
    train_df.drop('PassengerId',axis=1,inplace=1)
    test_df.drop('PassengerId',axis=1,inplace=1)
    features_list=train_df.columns.values[1:]
    X=train_df.values[:,1:]
    y=train_df.values[:,0]
    X_test=test_df.values
    ########################Step5: Reduce initial feature set with estimated feature importance
    print "Rough fitting a RandomForest to determine feature importance"
    forest=RandomForestClassifier(oob_score=True,n_estimators=10000,n_jobs=-1)
    forest.fit(X,y)
    feature_importance=forest.feature_importances_
    feature_importance=100.0*(feature_importance/feature_importance.max())
    print "Feature importances:\n", feature_importance
    fi_threshold=30
    important_idx=np.where(feature_importance>fi_threshold)[0]
    important_features=features_list[important_idx]
    print "\n", important_features.shape[0], "Important features(>", fi_threshold, "percent of max importance)...\n",important_features
    sorted_idx=np.argsort(feature_importance[important_idx])[::-1]
    #plot feature importance
    pos=np.arange(sorted_idx.shape[0])+0.5
    plt.subplot(1,2,2)
    plt.barh(pos,feature_importance[important_idx][sorted_idx[::-1]],align='center')
    plt.yticks(pos,important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.draw()
    plt.show()
    #Remove non-important features from the feature set
    X=X[:,important_idx][:,sorted_idx]
    X_test=X_test[:,important_idx][:,sorted_idx]
    print "\nSorted (DESC) Useful X:\n",X
    test_df=test_df.iloc[:,important_idx].iloc[:,sorted_idx]
    print '\nTraining with',X.shape[1],"features:\n",test_df.columns.values
    
    ########################Step6:Parameter tunning with CrossValidation(RandomSearch)###########
    ###Random search the best parameter
    """
    sqrtfeat=int(np.sqrt(X.shape[1]))
    params_test={"n_estimators":[10000],
                 "max_features":np.rint(np.linspace(sqrtfeat,sqrtfeat,3)).astype(int),
                 "min_samples_split":np.rint(np.linspace(X.shape[0]*0.01,X.shape[0]*0.2,30)).astype(int)}
    print "Hyperparameter opimization using RandomizedSearchCV..."
    rand_search=RandomizedSearchCV(forest,param_distributions=params_test,n_jobs=7,cv=3,n_iter=40)
    rand_search.fit(X,y)
    best_params=report(rand_search.grid_scores_)
    params=best_params
    """
    #==========================The best tunned parameters=========================================
    sqrtfeat=int(np.sqrt(X.shape[1]))
    minsampsplit=int(X.shape[0]*0.015)
    params_score={"n_estimators":10000,"max_features":sqrtfeat,"min_samples_split":minsampsplit}
    params=params_score
    #============================================================================================
    
    ########################Step7:Model generation/validation(Learning curve/Roc curve)#############
    print "Generating RandomForestClassifier model with parameters:",params
    forest=RandomForestClassifier(n_jobs=-1,oob_score=True,**params)
    print "\nCalculating Learning Curve..."
    print "\nCalculating  ROC curve ..."
    ###Calculate the OOB score
    print "\nCalculating the OOB score..."
    test_scores=[]
    for i in range(5):
        forest.fit(X,y)
        print "\nOOB:",forest.oob_score_
        test_scores.append(forest.oob_score_)
    oob="%.3f"%(np.mean(test_scores))
    oob_std="%.3f"%(np.std(test_scores))
    print "OOB mean score:",oob,"and stddev:",oob_std
    ########################Step8:Predicting and Saving result######################################
    submission=np.asarray(zip(test_ids,forest.predict(X_test))).astype(int)
    #ensure passenger IDs in ascending order
    output=submission[submission[:,0].argsort()]
    predict_file=open(path+"predict.csv",'wb')
    file_object=csv.writer(predict_file)
    file_object.writerow(["PassengerId","Survived"])
    file_object.writerows(output)
    predict_file.close()
    print 'Done'