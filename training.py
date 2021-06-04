import csv
import os
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import joblib
from sklearn.pipeline import Pipeline
from sentiment import TwitterClient
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation

#Generating the Training and testing vectors

twitterClient = TwitterClient()

def getTrainingAndTestData():
    
    dataset = pd.read_csv('final_dataset.csv', encoding = 'latin')
    dataset.drop(['id','date','username'], axis=1, inplace=True)
    dataset['target'] = dataset['target'].replace([0, 4],[0,1])
    print(dataset.shape)
    print(dataset.head())
    
    # print(dataset.target.unique())
    # print(dataset['target'].value_counts())

#     dataset_2=pd.read_csv('Dataset/dataset_2.csv', encoding='ISO-8859-1', header=None)
#     dataset_2 = dataset_2.rename(columns={0: 'target', 1: 'id', 2: 'date', 3: 'topic', 4: 'username', 5: 'content'})
#     dataset_2.drop(['id','date','topic','username'], axis=1, inplace=True)
#     dataset_2['target'] = dataset_2['target'].replace([0, 4],[0,1])
    
#     # print(dataset_2['target'].unique())
#     dataset_2 = dataset_2.drop(dataset_2[dataset_2['target'] == 2].index)
#     # print(dataset_2.head())
#     # print(dataset_2['target'].unique())
    
#     # print(dataset_2.shape)
    
#     merged_df = pd.concat([dataset, dataset_2])
#     # print(merged_df.shape)

#     merged_df.drop_duplicates(subset=['content'])
    
    X = dataset['content']
    y = dataset['target']

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=0.20, random_state=42)
    
    print("Created Dataset")
    return X_train, X_test, y_train, y_test

#Process Tweets (Stemming+Pre-processing)
def processTweets(X_train, X_test):
       
        print("Cleaning...")
        X_train = [twitterClient.stem(twitterClient.preprocessTweets(tweet)) for tweet in X_train]
        X_test = [twitterClient.stem(twitterClient.preprocessTweets(tweet)) for tweet in X_test]
        return X_train,X_test
        
# SVM classifier
def classifier(X_train,y_train):
        vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
        svm_clf =svm.LinearSVC(C=0.1)
        vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
        print("Training...")
        vec_clf.fit(X_train,y_train)
        joblib.dump(vec_clf, 'svmClassifier.pkl', compress=3)
        print("Saved Model")
        return vec_clf

# Main function

def main():
            
        X_train, X_test, y_train, y_test = getTrainingAndTestData()
        # X_train, X_test = processTweets(X_train, X_test)
        # vec_clf = classifier(X_train,y_train)
        # print("Metrics:")
        # y_pred = vec_clf.predict(X_test)
        # print(sklearn.metrics.classification_report(y_test, y_pred))
        
if __name__ == "__main__":
    main()