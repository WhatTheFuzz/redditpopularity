import sys
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import pandas
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import neighbors, datasets, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

classifiers = [
    SVC(kernel='linear', C=1),
    DecisionTreeClassifier(criterion='entropy', max_depth=100, random_state=0),
    GaussianNB()]
names = [
    "Linear SVM",
    "Decision Tree",
    "Naive Bayes"]

#We need to get the number of lines in the file to ensure we have enough comments for our training set.
#http://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if __name__ == "__main__":
    #where all of the csvs are stored.
    sub_path = "subreddit_csv"
    files = [f for f in listdir(sub_path) if isfile(join(sub_path, f)) if file_len(join(sub_path, f))>500]
    for f in files:
        data = pandas.read_csv(os.path.join(sub_path, f))

        popular_encoder = preprocessing.LabelEncoder()
        data.popular = popular_encoder.fit_transform(data.popular.tolist())
        #print("Popular encoder test: {0}".format(np.unique(popular_encoder.classes_) == [0, 1]))
        
        distinguished_encoder = preprocessing.LabelEncoder()
        data.distinguished = distinguished_encoder.fit_transform(data.distinguished.tolist())
        
        sentiment_encoder = preprocessing.LabelEncoder()
        data.sentiment = sentiment_encoder.fit_transform(data.sentiment.tolist())
        
        cv = CountVectorizer()
        list_of_body = data.body.as_matrix()
        bag = cv.fit_transform(list_of_body)
        word_freq = bag.toarray()
        #print(cv.inverse_transform(word_freq[0]))
        
        columns = ["distinguished", "sentiment", "created_utc", "controversiality"]
        features = data[list(columns)].values
        #features = np.hstack((data[list(columns)].values, word_freq))
        labels = data.popular.values
        
        X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=33)
        scaler = preprocessing.StandardScaler().fit(X_train)
        #X_train = scaler.transform(X_train)
        #X_test = scaler.transform(X_test)
        print "Testing subreddit: {0} with {1} comments.".format(f.replace(".csv", ""), file_len(join(sub_path, f)))
        for (name, clf) in zip(names, classifiers):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        #print X_train
            score = cross_val_score(clf, features, labels, n_jobs = -1, cv=4)
            print("{0} with five folds cross validation found an accuracy of: %0.3f (+/- %0.2f)".format(name) % (score.mean(), score.std() * 2))
        print "\n" 
