import sys
import os
import pandas as pd
import pickle

from lightfm import LightFM
from lightfm.datasets import fetch_movielens
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score

def create_model(train_data, loss_function, pickle_file):
  model = LightFM(learning_rate=0.05, loss=loss_function)
  model.fit(train_data, epochs=10, num_threads=4)
  pickle.dump(model, open(pickle_file, 'wb'))

def evaluate_model(train_data, test_data, pickle_file):
  model = pickle.load(open(pickle_file, 'rb'))
  train_precision = precision_at_k(model, train_data, k=10).mean()
  test_precision = precision_at_k(model, test_data, k=10).mean()

  train_auc = auc_score(model, train_data).mean()
  test_auc = auc_score(model, test_data).mean()

  print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
  print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

def create_embeddings():
  pass

def build_recommender(name='official'):

  if name=='official':
    data = fetch_movielens()
    train = data['train']
    test = data['test']
  else:
    data_path = './../datasets/'+name
    train_path = data_path+'/random_split/ratings_train.csv'
    test_path = data_path+'/random_split/ratings_train.csv'
    train, test = create_embeddings(data_path, train_path, test_path)

  loss_types= ['bpr', 'warp']

  for loss in loss_types:
    pickle_name = name+'_'+loss+'_random.pkl'
    create_model(train, loss, pickle_name)
    evaluate_model(train, test, pickle_name)


build_recommender()

# datasets = ['ml-100k', 'ml-1m', 'ml-10m', 'ml-20m']

# for dataset in datasets:
#   build_recommender(dataset)