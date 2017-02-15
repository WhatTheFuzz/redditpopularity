How to Get Upvotes on Reddit
============

![Team Spark](https://upload.wikimedia.org/wikipedia/en/thumb/8/82/Reddit_logo_and_wordmark.svg/1920px-Reddit_logo_and_wordmark.svg.png)

This project seeks to validate the results of previous authors' work, such as Tracy Rohlin's Master's Thesis for San Jose State in his work, ["Popularity Prediction of Reddit Texts."](http://scholarworks.sjsu.edu/etd_theses/4704/) As there are no courses at West Point dealing with machine learning, this project also introduces me to the field and will serve as my Honor's Theis. 

## Objective
- With the input of any particular Reddit comment, determine if the comment is likely to be popular within that subreddit.
- Achieve >75% predcitability. 

## Data
The data used in this project comes from r/datasets, in which one user posted [every publicly available Reddit comment](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/) up to early 2015. The data spans 1.7 billion comments. 

A single comments looks like so:
```
{"gilded":0,"author_flair_text":"Male","author_flair_css_class":"male","retrieved_on":1425124228,"ups":3,"subreddit_id":"t5_2s30g","edited":false,"controversiality":0,"parent_id":"t1_cnapn0k","subreddit":"AskMen","body":"I can't agree with passing the blame, but I'm glad to hear it's at least helping you with the anxiety. I went the other direction and started taking responsibility for everything. I had to realize that people make mistakes including myself and it's gonna be alright. I don't have to be shackled to my mistakes and I don't have to be afraid of making them. ","created_utc":"1420070668","downs":0,"score":3,"author":"TheDukeofEtown","archived":false,"distinguished":null,"id":"cnasd6x","score_hidden":false,"name":"t1_cnasd6x","link_id":"t3_2qyhmp"}
```
## Requirements
- This was tested solely on a 64-bit flavor of SuSE Enterprise Linux. YMMV.
- Python 2.7.10, or other compatible version.
- modules: nltk, sklearn, re, json, time, sys, and csv.

## Usage
- Clone this repository to a folder of your choosing.
- exectue: ```python preprocess.py dir_to_reddit_comments```, depending on how large the comment file is, this may take a non-trivial amount of time (several hours). This is a one-time process, as it will write the data into another file, which we will then analyze.
- once complete execute: ```python svm.py```. This will automatically look for the file "training.csv" and preform learn from it, eventually testing it against the test data and producing a result such as: ```Accuracy: 0.67 (+/- 0.07)``` (higher (~1) is better)

## Misc.
- In the preprocessing step, I opted for readability and compartmentalization over raw speed. Given that this only is run once before writing to a .csv, I felt it was worth the loss in speed. This is why you might notice that I loop through each comment multiple times. 
- For information about the author, please visit his website at [www.seandeaton.com](https://www.seandeaton.com).
