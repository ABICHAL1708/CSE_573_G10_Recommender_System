import pandas as pd
import os
import csv
import requests
import json
from itertools import islice

import lightfm
from lightfm.data import Dataset


# Adding the base dir
base_path = "./.."

# Setting constant seed for reproducibility
SEED = 42

def split_data(data_dir):
	print("===========================================")
	print("Splitting the Data")
	print("===========================================")
	data_dir = base_path+"/"+data_dir

	print("Data files are:")
	print(os.listdir(data_dir))

	print("Loading and splitting the ratings.csv file")

	# Load ratings.csv
	rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
	ratings = pd.read_table(data_dir+"/ratings.dat", sep = "::", header = None, names = rating_columns, encoding = "latin-1")

	# Splitting the data
	# Creating train data
	ratings_train = ratings.sample(frac=0.8, random_state = SEED)

	# Dropping the train from ratings to get test
	ratings_test = ratings.drop(ratings_train.index)

	# Checking shapes of train and test
	print("Train test split completed")
	print("___________________________")
	print("train shape:")
	print(ratings_train.shape)
	print("___________________________")
	print("test shape:")
	print(ratings_test.shape)
	print("___________________________")

	ratings_train.to_csv(data_dir+"/random_split/ratings_train.csv")
	ratings_test.to_csv(data_dir+"/random_split/ratings_test.csv")


data_dir = "datasets/ml-1m"
split_data(data_dir)