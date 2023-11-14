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
# print(official_movielens_data["item_features"])

# Setting the official dataset here
movielens_data = official_movielens_data

# Check the dataset
print("The movielens datset loaded is a dict with the following keys:")
# print(movielens_data.keys())
# print(movielens_data["train"])

train = movielens_data['train']
test = movielens_data['test']
item_features = movielens_data['item_features']

# Instantiating the model on BPR loss
model = LightFM(learning_rate=0.1, loss='warp')
model.fit(train, epochs=30, item_features = item_features)

train_precision = precision_at_k(model, train, k=10).mean()
test_precision = precision_at_k(model, test, k=10).mean()

train_auc = auc_score(model, train).mean()
test_auc = auc_score(model, test).mean()

print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))