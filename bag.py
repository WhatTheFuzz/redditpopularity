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
from sklearn.metrics import accuracy_score, cohen_kappa_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import tree
from IPython.display import Image
import graphviz
import pydotplus

classifiers = [
    DecisionTreeClassifier(criterion='entropy', random_state=0)]
words_classifiers = copy.deepcopy(classifiers)
names = [
    "Decision Tree"]

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
        
        cv = CountVectorizer()
        list_of_body = data.body.as_matrix()
        bag = cv.fit_transform(list_of_body)
        word_freq = bag.toarray()
        
        features =  word_freq
        labels = data.popular.values

        X_train, X_test, y_train, y_test = train_test_split(features, labels, random_state=random.randint(1,10000))
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        file = open("results/bag.dat","a+")
        print("Testing subreddit: {0} with {1} comments.\n".format(f.replace(".csv", ""), file_len(join(sub_path, f))))
        for (name, clf) in zip(names, classifiers):
            clf.fit(X_train, y_train)
            y_pred = cross_val_predict(clf, features, labels, n_jobs=-1, cv=10)
            
            score = cross_val_score(clf, features, labels, n_jobs=-1, cv=10)
            if (name=="Decision Tree"):
                dot_data = tree.export_graphviz(clf, out_file=None,
                                                feature_names = map(lambda x: str(x), cv.get_feature_names()),
                                                filled=True, rounded=True,
                                                special_characters=False)
                graph = pydotplus.graph_from_dot_data(dot_data)
                graph.write_pdf("results/pdf/bag_{0}.pdf".format(f.replace(".csv", "")))
            print("{0} with five folds cross validation found an accuracy of: %0.3f (+/- %0.2f)\n".format(name) % (score.mean(), score.std() * 2))
            print("Cohen Kappa Score: %0.3f\n" % (cohen_kappa_score(y_pred, labels))) 
        print("\n") 
