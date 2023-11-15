import pandas as pd
import os
import csv
import requests
import json
import itertools
from itertools import islice
import scipy
from scipy import sparse
import json

# Lightfm libraries
import lightfm
from lightfm.data import Dataset


# Adding the base dir
base_path = ""

# Setting constant seed for reproducibility
SEED = 42

def create_features(data_dir):
	print("===========================================")
	print("Loading and Preprocessing the Data")
	print("===========================================")
	# data_dir = base_path+"/"+data_dir

	# Preprocessing for Datset
	# Creating Ratings, Movies and Users dictreader
	print("Preprocessing for dataset")

	print("Loading ratings, movies and users")
	ratings = pd.read_csv(data_dir+"/ratings.csv")
	movies = pd.read_csv(data_dir+"/movies.csv")
	# users = pd.read_csv(data_dir+"/users.csv")

	# Drop first column
	print("Dropping the first columns")
	ratings = ratings.drop(columns = ["Unnamed: 0"])
	movies = movies.drop(columns = ["Unnamed: 0"])
	# users = users.drop(columns = ["Unnamed: 0"])


	print("===========================================")
	print("Creating Features")
	print("===========================================")
	# Adding genres and occupations columns to data
	data = ratings
	print("Merging data with genres and occupations columns:")
	data = data.merge(movies[['movie_id', 'genres']], left_on='movie_id', right_on='movie_id')

	# data = data.merge(users[['user_id', 'occupation']], left_on='user_id', right_on='user_id')

	print("Checking Random Samples of merged data:")
	print(data.sample(5, random_state=42))

	# Getting all the genres for movie ids in data
	movie_genre = [x.split('|') for x in data["genres"]]	

	# Listing all unique genres and Occupation
	movies_genre_list = movies["genres"].tolist()
	movies_genre_dict = {}

	for genres in movies_genre_list:
		genres = genres.split("|")
		for genre in genres:
			movies_genre_dict[genre] = 0

	all_movie_genre = sorted(list(movies_genre_dict.keys()))
	# all_occupations = sorted(list(set(users['occupation'])))

	# Creating Dataset object and fitting
	dataset = Dataset()
	# dataset.fit(data['user_id'], data['movie_id'], item_features = all_movie_genre, user_features = all_occupations)
	dataset.fit(data['user_id'], data['movie_id'], item_features = all_movie_genre)

	# Retreiving all Item and User features
	item_features = dataset.build_item_features((x, y) for x,y in zip(data.movie_id, movie_genre))

	# user_features = dataset.build_user_features((x, [y]) for x,y in zip(data['user_id'], data['occupation']))

	# Check item and user features
	print("Type of Item and User Features:")
	print(type(item_features))
	# print(type(user_features))

	# Saving the item and user feature matrices
	sparse.save_npz(data_dir+"/item_features.npz", item_features)
	# sparse.save_npz(data_dir+"/user_features.npz", user_features)

	print("===========================================")
	print("Creating Interaction Matrix")
	print("===========================================")
	# Using the first 3 columns for 
	print(data.iloc[:, 0:3])

	interactions, weights2 = dataset.build_interactions(data.iloc[:, 0:3].values)

	print("Type of Interactions")
	print(type(interactions))
	sparse.save_npz(data_dir+"/interactions.npz", interactions)

	print("===========================================")
	print("Checking User/Movie Mappings")
	print("===========================================")
	print("Getting the mappings")
	uid_map, ufeature_map, iid_map, ifeature_map = dataset.mapping()

	print("The type of uid_map is: ")
	print(type(uid_map))

	print("Saving the uid, iid, and ifeatures mappings")

	with open(data_dir+"/uid_map.json", "w") as outfile:
		json.dump(uid_map, outfile)
	with open(data_dir+"/iid_map.json", "w") as outfile:
		json.dump(iid_map, outfile)
	with open(data_dir+"/ifeature_map.json", "w") as outfile:
		json.dump(ifeature_map, outfile)


def create_csv(data_dir):
	print("===========================================")
	print("Saving the Data")
	print("===========================================")
	# data_dir = base_path+"/"+data_dir

	print("Data files are:")
	print(os.listdir(data_dir))

	# Check if movies.csv, ratings.csv and users.csv already present
	file_list = os.listdir(data_dir)
	if("movies.csv" not in file_list):
		if("movies.dat" in file_list):
			# Load Movies.dat
			movie_columns = ['movie_id', 'title', 'genres']
			movies = pd.read_table(data_dir+"/movies.dat", sep = "::", header = None, names = movie_columns, encoding = "latin-1")

			# Save movies data to csv
			movies.to_csv(data_dir+"/movies.csv")
		else:
			print("movies.dat not found")

	if("ratings.csv" not in file_list):
		if("ratings.dat" in file_list):
			# Load Ratings.dat
			rating_columns = ['user_id', 'movie_id', 'rating', 'timestamp']
			ratings = pd.read_table(data_dir+"/ratings.dat", sep = "::", header = None, names = rating_columns, encoding = "latin-1")

			# Save ratings to csv
			ratings.to_csv(data_dir+"/ratings.csv")
		else:
			print("ratings.dat not found")

	if("users.csv" not in file_list):
		if("users.dat" in file_list):
			# Load users.dat
			user_columns = ['user_id', 'gender', 'age', 'occupation', 'zip']
			users = pd.read_table(data_dir+"/users.dat", sep = "::", header = None, names = user_columns, encoding = "latin-1")

			# Save users to csv
			users.to_csv(data_dir+"/users.csv")
		else:
			print("users.dat not found")

# # 1m
# data_dir = "datasets/ml-1m"
# create_csv(data_dir)
# create_features(data_dir)

# # 10m
# data_dir = "datasets/ml-10m"
# create_csv(data_dir)
# create_features(data_dir)

# # 20m
# data_dir = "datasets/ml-20m"
# create_csv(data_dir)
# create_features(data_dir)

# ml-demo
data_dir = "datasets/ml-demo"
# create_csv(data_dir)
create_features(data_dir)