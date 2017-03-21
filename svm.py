import sys
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import sklearn
import pandas
import os
from os import listdir
from os.path import isfile, join

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
    files = [f for f in listdir(sub_path) if isfile(join(sub_path, f)) if file_len(join(sub_path, f))>10]
    for f in files:
        #data = np.genfromtxt("training.csv", delimiter = ",", dtype = {'names': ("ups", "subreddit", "popular"), "formats": (np.int, "|S20", np.bool)})
        data = pandas.read_csv(os.path.join(sub_path, f))
        #init our label encoder to turn the subreddit and popular strings/bools into categories
        encoder = preprocessing.LabelEncoder()
        #init our vectorizer, which will transform our strings into a matrix
        cv = CountVectorizer()
        #list_to_vectorize = (list(set(data.body.tolist())))
        
        #data.subreddit = encoder.fit_transform(data.subreddit)
        #subs = encoder.fit(data.subreddit)
        data.subreddit = encoder.fit_transform(data.subreddit.tolist())
        data.popular = encoder.fit_transform(data.popular.tolist())
        #data.distinguished = encoder.fit_transform(data.popular.tolist())
        list_of_body = data.body.as_matrix()
        bag = cv.fit_transform(list_of_body)
        data.body = bag.toarray()
        #np.set_printoptions(threshold='nan')
        '''
        #ensure our vocab is correct - if we transform additional elements, we should get the number of samples mapped to the feature
        print(cv.vocabulary_)
        new_data = cv.transform(["1930s", "heart"])
        print new_data
        '''
        #columns = features we want to fit. labels = what we want to predict
        #columns= ["subreddit", "ups", "controversiality", "created_utc", "body"]
        columns= ["subreddit", "created_utc", "controversiality"]
        labels = data["popular"].values
        binary_features = data[list(columns)].values
        wordfreq_features = bag.toarray()
        final_features = (np.hstack((binary_features, wordfreq_features)))
        print "Testing subreddit: {0}".format(f.replace(".csv", ""))
        #print "Label length is: {0}. Feature length is: {1}. These should match.".format(len(labels), (len(final_features)))
        #print "The features include: {}".format(", ".join(columns))
        
        #print "training labels: {0}".format(labels)
        #print "training features: {0}".format(features)
        
        X_train, X_test, y_train, y_test = train_test_split(
        final_features, labels, test_size=0.4, random_state=0)
        clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
        print("Partitioning with 60/40 gave the result: %0.2f" % clf.score(X_test, y_test))
        
        classifier = SVC()
        classifier.fit(final_features, labels)
        #n_jobs = # of cpus, with 'cv' folds.
        score = cross_val_score(classifier, final_features, labels, n_jobs = -1, cv=4)
        print("With five folds cross validation found an accuracy of: %0.2f (+/- %0.2f)\n" % (score.mean(), score.std() * 2))
