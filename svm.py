import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import sklearn
import pandas


if __name__ == "__main__":
    #data = np.genfromtxt("training.csv", delimiter = ",", dtype = {'names': ("ups", "subreddit", "popular"), "formats": (np.int, "|S20", np.bool)})
    data = pandas.read_csv("training.csv")
    encoder = preprocessing.LabelEncoder()
    #for i in range(3):
    #    data[:,i] = encoder.fit_transform(data[:,i])
    data.subreddit = encoder.fit_transform(data.subreddit)
    data.popular = encoder.fit_transform(data.popular)

    columns = ["subreddit", "ups"]
    labels = data["popular"].values
    features = data[list(columns)].values

    print "training labels: {0}".format(labels)
    print "training features: {0}".format(features)

    classifier = SVC()
    classifier.fit(features, labels)
    score = cross_val_score(classifier, features, labels, n_jobs = 1)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
    '''
    labels = list(set(train_labels))
    train_labels = np.array([labels.index(x) for x in train_labels])
    train_features = data.iloc[:,1:]
    train_features = np.array(train_features)

    print "train labels: {0}".format(train_labels)
    print "train features: {0}".format(train_features)

    classifier = SVC()
    classifier.fit(train_features, train_labels)

    results = classifier.predict(train_features)
    num_correct = (results == train_labels).sum()
    print(num_correct)
    '''
