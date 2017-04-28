import sys
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import pandas
import random
import copy

from sklearn import neighbors, datasets, preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, cohen_kappa_score, \
                            precision_recall_curve, average_precision_score, recall_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

classifiers = [
    SVC(kernel='linear', C=1),
    #must set max-depth or we run into memory errors.
    DecisionTreeClassifier(criterion='entropy', random_state=0),
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
    sub_path = "subreddit_twomillion_csv"
    files = [f for f in listdir(sub_path) if isfile(join(sub_path, f)) if file_len(join(sub_path, f))>1750]
    for f in files:
        data = pandas.read_csv(os.path.join(sub_path, f))

        popular_encoder = preprocessing.LabelEncoder()
        data.popular = popular_encoder.fit_transform(data.popular.tolist())
        #file.write("Popular encoder test: {0}".format(np.unique(popular_encoder.classes_) == [0, 1]))
        
        distinguished_encoder = preprocessing.LabelEncoder()
        data.distinguished = distinguished_encoder.fit_transform(data.distinguished.tolist())
        
        sentiment_encoder = preprocessing.LabelEncoder()
        data.sentiment = sentiment_encoder.fit_transform(data.sentiment.tolist())
        
        cv = CountVectorizer()
        list_of_body = data.body.as_matrix()
        bag = cv.fit_transform(list_of_body)
        word_freq = bag.toarray()
        
        columns = ["distinguished", "sentiment", "created_utc", "controversiality"]
        features = np.hstack((data[list(columns)].values, word_freq))
        labels = data.popular.values

        X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=random.randint(1,10000))
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        file = open("results/clf_results_1750.dat","a+")
        file.write("\nTesting subreddit: {0} with {1} comments.\n".format(f.replace(".csv", ""), file_len(join(sub_path, f))))
        for (name, clf) in zip(names, classifiers):
            clf.fit(X_train, y_train)
            y_pred = clf.predict(features)
            score = cross_val_score(clf, features, labels, n_jobs=-1, cv=10)

            precision, recall, thresholds = precision_recall_curve(labels, y_pred)
            acc_score = accuracy_score(labels, y_pred)
            avg_precision = average_precision_score(labels, y_pred)
            avg_recall = recall_score(labels, y_pred)

            file.write("{0}:\nScore: %0.3f (+/- %0.2f)\n".format(name) % (score.mean(), score.std() * 2))
            file.write("Cohen Kappa Score: %0.3f\n" % (cohen_kappa_score(y_pred, labels)))
            file.write("Accuracy:%0.3f, Precision:%0.3f, Recall:%0.3f\n" % (acc_score, avg_precision, avg_recall))
