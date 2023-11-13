import sys
import os
import pandas as pd
import scipy
from scipy import sparse

# Import the model
from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.data import Dataset

# Adding the base dir as path
base_path = "/Users/abichalghosh/Documents/1-3/SWM/Project"

# Load the MovieLens 100k dataset using lightfm's official code
official_movielens_data = fetch_movielens()

print(official_movielens_data.keys())
print(official_movielens_data["item_features"])

import os
import zipfile
import csv
import requests
import json
from itertools import islice
from lightfm.data import Dataset


def create_features(data_dir):
	ratings = csv.DictReader(x for x in open("ratings.csv"))
	movie_features = csv.DictReader(x for x in open("books.csv"))

def preprocess_data(data_dir):
	# EDIT
	data_dir = base_path+"/"+data_dir
	print("===========================================")
	print("Preprocessing the Data")
	print("===========================================")
	print("The files present in the data directory are given as-")
	print(os.listdir(data_dir))

	movie_columns = ['movie_id', 'title', 'genres']
	movies = pd.read_table(data_dir+"/movies.dat", sep = "::", header = None, names = movie_columns, encoding = "latin-1")

	rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
	ratings = pd.read_table(data_dir+"/ratings.dat", sep = "::", header = None, names = rating_columns, encoding = "latin-1")

	user_columns = ['user_id', 'gender', 'age', 'occupation', 'zip']
	users = pd.read_table(data_dir+"/users.dat", sep = "::", header = None, names = user_columns, encoding = "latin-1")

	# Preprocessing the ratings 
	print(ratings)

	# Creating the dataset object
	dataset = Dataset()

	
	return " "

# Loading the downloaded movielens dataset
data_dir = "datasets/movielens_dataset/ml-1m"
movielens_data = preprocess_data(data_dir)

# Setting the official dataset here
movielens_data = official_movielens_data

# Check the dataset
print("The movielens datset loaded is a dict with the following keys:")
# print(movielens_data.keys())
# print(movielens_data["train"])

train = movielens_data['train']
test = movielens_data['test']

# Instantiating the model on BPR loss
model = LightFM(learning_rate=0.05, loss='bpr')
model.fit(train, epochs=10)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))