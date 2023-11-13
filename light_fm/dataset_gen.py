import pandas as pd
import os
import csv
import requests
import json
from itertools import islice
import scipy
from scipy import sparse

# Lightfm libraries
import lightfm
from lightfm.data import Dataset


# Adding the base dir
base_path = "./.."

# Setting constant seed for reproducibility
SEED = 42

def create_features(data_dir):
	pass

def create_csv(data_dir):
	print("===========================================")
	print("Splitting the Data")
	print("===========================================")
	data_dir = base_path+"/"+data_dir

	print("Data files are:")
	print(os.listdir(data_dir))

	# Check if movies.csv, ratings.csv and users.csv already present
	file_list = os.listdir(data_dir)
	if("movies.csv" not in file_list):
		# Load Movies.dat
		movie_columns = ['movie_id', 'title', 'genres']
		movies = pd.read_table(data_dir+"/movies.dat", sep = "::", header = None, names = movie_columns, encoding = "latin-1")

		# Save movies data to csv
		movies.to_csv(data_dir+"/movies.csv")

	if("ratings.csv" not in file_list):
		# Load Ratings.dat
		rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
		ratings = pd.read_table(data_dir+"/ratings.dat", sep = "::", header = None, names = rating_columns, encoding = "latin-1")

		# Save ratings to csv
		ratings.to_csv(data_dir+"/ratings.csv")

	if("users.csv" not in file_list):
		# Load users.dat
		user_columns = ['user_id', 'gender', 'age', 'occupation', 'zip']
		users = pd.read_table(data_dir+"/users.dat", sep = "::", header = None, names = user_columns, encoding = "latin-1")

		# Save users to csv
		users.to_csv(data_dir+"/users.csv")


data_dir = "datasets/ml-1m"
create_csv(data_dir)