import sys
import os
import pandas as pd
import pickle
import numpy as np

from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import reciprocal_rank
from scipy import sparse


SEED = 42
epochs = 34
learning_rate = 0.221

def create_model(train_data, loss_function, item_features, pickle_file):
	model = LightFM(learning_rate=learning_rate, loss=loss_function, random_state=np.random.RandomState(SEED))
	model.fit(train_data, item_features=item_features, epochs=epochs, verbose=True)
	pickle.dump(model, open(pickle_file, 'wb'))

def evaluate_model(train_data, test_data, item_features, pickle_file):
	model = pickle.load(open(pickle_file, 'rb'))
	train_precision = precision_at_k(model, train_data, item_features=item_features).mean()
	test_precision = precision_at_k(model, test_data, item_features=item_features).mean()

	train_auc = auc_score(model, train_data, item_features=item_features).mean()
	test_auc = auc_score(model, test_data, item_features=item_features).mean()

	train_recall = recall_at_k(model, train_data, item_features=item_features).mean()
	test_recall = recall_at_k(model, test_data, item_features=item_features).mean()

	train_rank = reciprocal_rank(model, train_data, item_features=item_features).mean()
	test_rank = reciprocal_rank(model, test_data, item_features=item_features).mean()

	print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
	print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
	print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
	print('Rank: train %.2f, test %.2f.' % (train_rank, test_rank))

def build_recommender(name):
	data = sparse.load_npz("interactions.npz")
	item_features = sparse.load_npz("item_features.npz")
	train, test = random_train_test_split(data, random_state=np.random.RandomState(SEED))

	# loss_types= ['bpr', 'warp']
	loss_types = ['warp']

	for loss in loss_types:
		pickle_name = '{0}_{1}_random.pkl'.format(name, loss)
		create_model(train, loss, item_features, pickle_name)
		evaluate_model(train, test, item_features, pickle_name)

# datasets = ['100K', '1M', '10M', '20M']

# for dataset in datasets:
# 	build_recommender(dataset)
build_recommender('1M')