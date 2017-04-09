from __future__ import absolute_import
from __future__ import print_function
import csv
import sys
import time
import json
import re
import os
import datetime
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#from textblob.classifiers import NaiveBayesClassifier
from textblob import classifiers
from pyspark import SparkContext

def read_JSON_as_dict(file_name):
    #read the file in as a string and remove non-alphanumeric characters
    dict_of_subs = {}
    for line in open(file_name, 'r'):
        json_object=(json.loads(line.lower()))
        subreddit = json_object["subreddit"]
        #the key is the subreddit and the value is the entirely of the dictionary. Thus we will have a list of dictionaries.
        dict_of_subs.setdefault(subreddit, []).append(json_object)
    return dict_of_subs

def remove_function_words(dict_of_subs):
    dict_of_all_words = {}
    for sub, list_of_comments in dict_of_subs.iteritems():
        for comment in list_of_comments:
            comment_list = wordList = re.sub("[^\w]", " ",  comment["body"]).split()
            comment["body"] = comment_list
            for word in comment_list:
                dict_of_all_words[word] = dict_of_all_words.get(word, 0) + 1
            comment["body"] = ([word for word in comment_list if word not in stopwords.words("english")])
    return dict_of_subs

def filter_votes_length(dict_of_subs):
    dict = dict_of_subs
    for sub, list_of_comments in dict.iteritems():
        dict[sub] = [comment for comment in list_of_comments if int(comment["ups"]) >= 2 if len(comment["body"]) >= 20]
        #dict["body"] = " ".join(comment["body"])
    return dict
'''
def split_training_test(dict_of_subs):
    comment_number = 0
    training_dict = {}
    testing_dict = {}
    for sub, list_of_comments in dict_of_subs.iteritems():
        for comment in list_of_comments:
            if (comment_number % 8 == 0): testing_dict.setdefault(sub, []).append(comment)
            else: training_dict.setdefault(sub, []).append(comment)
            comment_number +=1
    training_dict = {k: v for k, v in training_dict.iteritems() if v}
    testing_dict = {k: v for k, v in testing_dict.iteritems() if v}
    return (training_dict, testing_dict)
'''

def determine_if_popular(training_dict):
    dict_of_upvotes = {} 
    for sub, list_of_comments in training_dict.iteritems():
        sub_list = []
        for comment in list_of_comments:
            sub_list.append(int(comment["ups"]))
        sorted_list = sorted(sub_list)
        dict_of_upvotes[sub] = int(sorted_list[(len(sorted_list)/4)*3])
    for sub, list_of_comments in training_dict.iteritems():
        for comment in list_of_comments:
            comment["body"] = " ".join(comment["body"])
            comment["created_utc"] = datetime.datetime.fromtimestamp(int(comment["created_utc"])).strftime("%H")
            comment["sentiment"] = str(sentiment_clf.classify(comment["body"]))
            if comment["distinguished"]!="":
                comment["distinguished"] = "not_distinguished"
            if (int(comment["ups"]) >= dict_of_upvotes[sub]): 
                comment["popular"] = True
            else: comment["popular"] = False
    return training_dict

def get_sentiment(training_dict):
    for sub, list_of_comments in training_dict.iteritems():
        for comment in list_of_comments:
            comment["sentiment"] = 0

def write_csv(file_name, training_list_of_comments):
    headers = ['ups', 'popular', 'subreddit', 'body', 'controversiality', 'created_utc', 'distinguished', 'sentiment']
    with open(os.path.join('til_csv', file_name), "wb") as file:
        w = csv.DictWriter(file, fieldnames=headers, extrasaction='ignore')
        w.writeheader()
        for comment in training_list_of_comments:
            w.writerow(comment)
        file.close()

def merge_dicts(a, b):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    dict_a = json.loads(b)
    return dict_a["subreddit"] 
 

lexicon = "lexicon.txt"
lexicon_csv = csv.reader(open(lexicon, "rb"), delimiter=" ")
sentiment_list = [(l[2].replace("word1=", ""), l[5].replace("priorpolarity=", "")) for l in lexicon_csv]

print("training the sentiment classifier")
clf_timer = time.time()
sentiment_clf = classifiers.NaiveBayesClassifier(sentiment_list)
print("It took {0}s to train the sentiement classifier".format(time.time() - clf_timer))

if __name__ == "__main__":
    sc = SparkContext(appName="PreprocessingCSV")
    sc.setLogLevel("WARN")

    file_name = str(sys.argv[1]) if len(sys.argv) > 1 else ("logs/RC_2015-01")
    number_of_partitions = int(sys.argv[2]) if len(sys.argv) > 2 else 36 #36=number of cores per node
    file = sc.textFile(file_name)#.repartition(number_of_partitions).cache()
    file.setName("Preproccessed_CSV")

    start_time = time.time()
    dict = read_JSON_as_dict(file_name) 
        #.map(lambda dict: {dict["subreddit"]:list(dict)})\
        #.reduce(lambda a, b: a.setdefault(b.keys()[0], [])\
        #    .append(b[b.keys()[0]][0]))
    #print type(dicts.take(10)[0])
    #dict_of_subs = file.reduce(lambda a, b: merge_dicts(a, b))
    '''
    dict_of_subs = read_JSON_as_dict(file_name)
    print("It took {0} to read the file.".format(time.time() - start_time))
    
    dict_of_subs_no_function_words = remove_function_words(dict_of_subs)
    dict_of_subs_stripped = filter_votes_length(dict_of_subs_no_function_words)
    dict_reduced = dict((k, v) for (k, v) in dict_of_subs_stripped.iteritems() if v)
    
    dict_tagged = determine_if_popular(dict_reduced)
    for k, v in dict_tagged.iteritems():
        write_csv(str(k)+".csv", v)
    print("total time: {0}.".format(time.time() - start_time))
    '''
