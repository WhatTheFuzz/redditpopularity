import sys
sys.path.append("/p/home/fourmore/python_modules/numpy/gnu/python2.7/lib/python2.7/site-packages")
import numpy as np


if __name__ == "__main__":
    data = np.genfromtxt("training.csv", delimiter = ",", dtype = {'names': ("ups", "subreddit", "popular"), "formats": (np.int, "|S20", np.bool)})


    print(data)
