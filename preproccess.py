import json


reddit_data = 'smallJSON'
json_file = open(reddit_data)
json_string = json_file.read()
dict_of_subs = {}
for line in open(reddit_data, 'r'):
    json_object=(json.loads(line.lower()))
    subreddit = json_object["subreddit"]
    dict_of_subs.setdefault(subreddit, []).append(json_object)
print dict_of_subs.keys()
