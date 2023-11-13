import sys
import os
import pandas as pd
import pickle

from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.evaluation import recall_at_k
from lightfm.evaluation import reciprocal_rank
from spotlight.datasets import movielens


def create_model(train_data, loss_function, pickle_file):
  model = LightFM(learning_rate=0.05, loss=loss_function)
  %time model.fit(train_data, epochs=10, num_threads=4, verbose=True)
  pickle.dump(model, open(pickle_file, 'wb'))

def evaluate_model(train_data, test_data, pickle_file):
  model = pickle.load(open(pickle_file, 'rb'))
  train_precision = precision_at_k(model, train_data).mean()
  test_precision = precision_at_k(model, test_data).mean()

  train_auc = auc_score(model, train_data).mean()
  test_auc = auc_score(model, test_data).mean()

  train_recall = recall_at_k(model, train_data).mean()
  test_recall = recall_at_k(model, test_data).mean()

  train_rank = reciprocal_rank(model, train_data).mean()
  test_rank = reciprocal_rank(model, test_data).mean()

  print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
  print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))
  print('Recall: train %.2f, test %.2f.' % (train_recall, test_recall))
  print('Rank: train %.2f, test %.2f.' % (train_rank, test_rank))

def create_embeddings():
  pass

def build_recommender(name):

  data = movielens.get_movielens_dataset(variant=name).tocoo()
  train, test = random_train_test_split(data)

  # loss_types= ['bpr', 'warp']
  loss_types = ['warp']

  for loss in loss_types:
    pickle_name = name+'_'+loss+'_random.pkl'
    create_model(train, loss, pickle_name)
    evaluate_model(train, test, pickle_name)

datasets = ['100K', '1M', '10M', '20M']

for dataset in datasets:
  build_recommender(dataset)