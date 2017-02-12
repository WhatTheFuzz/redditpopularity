import praw
import random

# Python Reddit API Wrapper: pip install praw
# http://praw.readthedocs.io/en/latest/getting_started/quick_start.html
# 
# I had to register an API application on Reddit to get my client id and client secret

reddit = praw.Reddit(client_id = "5nO1_wsqioGwBA",
                     client_secret = "PDxVwg951yJfNy5kAQ9wkcgeHk0",
                     user_agent = "user_agent_" + str(random.random()))

print(reddit.read_only)

subreddit = reddit.subreddit("news")

print("display name is: ", subreddit.display_name)
print("title is:", subreddit.title)
#print("description is: ", subreddit.description)

count = 1

# In Windows, issue the following command before running the script:
# chcp 65001 
# to allow Windows to support displaying the unicode charset

# this just grabs the top 10 stories from "hot" from the "news" subreddit.
# can also do limit=None, which pulls down 100 at a time until all the "hot" posts are pulled...
# Other options are:
# controversial
# gilded
# hot
# new
# rising
# top

for submission in subreddit.hot(limit=10):
    print("Submission", count, ":", submission.title)

# I was having to do this to avoid unprintable unicode characters from erroring, but chcp 65001 seems to make this unnecessary.
#    print("Submission", count, ":", end="")
#    for char in submission.title:
#        try:
#            print(char, end="")
#        except:
#            pass
#    print()

    count += 1
    print(vars(submission))

    print("\n")
    