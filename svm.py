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
    #init our label encoder to turn the subreddit and popular strings/bools into categories
    encoder = preprocessing.LabelEncoder()
    #init our vectorizer, which will transform our strings into a matrix
    cv = CountVectorizer(binary="true")
    list_to_vectorize = (list(set(data.body.tolist())))
    
    data.subreddit = encoder.fit_transform(data.subreddit)
    data.popular = encoder.fit_transform(data.popular)
    #turn into matrix, then a list that the SVC can fit. 
    data.body = cv.fit_transform(data.body).toarray()
    '''
    ensure our vocab is correct - if we transform additional elements, we should get the number of samples mapped to the feature
    print(cv.vocabulary_)
    new_data = cv.transform(["1930s", "heart"])
    print new_data
    '''
    #columns = features we want to fit. labels = what we want to predict
    columns= ["ups", "subreddit", "body"]
    labels = data["popular"].values
    features = data[list(columns)].values
    #print data["body"].values

    print "training labels: {0}".format(labels)
    print "training features: {0}".format(features)

    classifier = SVC()
    classifier.fit(features, labels)
    #n_jobs = # of cpus
    score = cross_val_score(classifier, features, labels, n_jobs = -1)
    print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
