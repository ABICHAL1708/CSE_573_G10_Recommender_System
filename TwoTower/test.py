from datetime import datetime
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import gensim.downloader
from ast import literal_eval
from collections import Counter, OrderedDict

import tensorflow_recommenders as tfrs
from gensim.parsing.preprocessing import strip_multiple_whitespaces, remove_stopwords, split_alphanum, strip_non_alphanum, strip_punctuation

import datetime

word2vec = gensim.downloader.load('glove-twitter-25')

with open('metadata.json', 'r') as f:
    metadata = json.load(f)

all_users = [str(int(i)) for i in metadata['users']]
all_movies = [str(int(i)) for i in metadata['movies']]
all_cities = metadata['cities']
all_states = metadata['states']
all_ages = [str(int(i)) for i in metadata['ages']]
all_occupations = [str(int(i)) for i in metadata['occupations']]
all_genres = metadata['genres']
title_emb_len = metadata['title_emb_size']
na_value = metadata['string_na']


ratings = pd.read_csv("data/ratings.csv")
users = pd.read_csv("data/users.csv")
movies = pd.read_csv("data/movies.csv")


movies[['title', 'movie_year', 'genres']] = movies.apply(lambda row: pd.Series({
    'title': row['title'][:-7],
    'movie_year': row['title'][-5:-1],
    'genres': str(row['genres']).split('|') if not pd.isnull(row['genres']) else list()
}), axis=1)


def user_features(row):
    zip_code = int(str(row['zip'])[:5]) if not pd.isnull(row['zip']) else 0
    try:
        z = zip_code_search.by_zipcode(zip_code).to_dict()
        return pd.Series({
            'gender': int(row['gender'] == 'F'),
            'city': z.get('major_city', ''),
            'state': z.get('state', ''),
            'zip': zip_code
        })
    except:
        return pd.Series({
            'gender': int(row['gender'] == 'F'),
            'city': 'XX',
            'state': 'XX',
            'zip': zip_code
        })

users[['gender', 'city', 'state', 'zip']] = users.apply(user_features, axis=1)



class RatingPredictionModel(tfrs.models.Model):
    def __init__(self):
        super().__init__()
        
        tower_last_layer_size = 50
        large_embedding_size = 25
        medium_embedding_size = 5
        small_embedding_size = 3
        
        # User tower
        
        self.user_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='user_input')
        self.user_sl = tf.keras.layers.StringLookup(vocabulary=all_users, name='user_string_lookup')(self.user_input)
        self.user_emb = tf.squeeze(tf.keras.layers.Embedding(len(all_users)+1, large_embedding_size, name='user_emb')(self.user_sl), axis=1)
        
        self.city_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='city_input')
        self.city_sl = tf.keras.layers.StringLookup(vocabulary=all_cities, mask_token=na_value, name='city_string_lookup')(self.city_input)
        self.city_emb = tf.squeeze(tf.keras.layers.Embedding(len(all_cities)+2, medium_embedding_size, name='city_emb')(self.city_sl), axis=1)
        
        self.state_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='state_input')
        self.state_sl = tf.keras.layers.StringLookup(vocabulary=all_states, mask_token=na_value, name='state_string_lookup')(self.state_input)
        self.state_emb = tf.squeeze(tf.keras.layers.Embedding(len(all_states)+2, small_embedding_size, name='state_emb')(self.state_sl), axis=1)
        
        self.age_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='age_input')
        self.age_sl = tf.keras.layers.StringLookup(vocabulary=all_ages, num_oov_indices=0, name='age_string_lookup')(self.age_input)
        self.age_emb = tf.squeeze(tf.keras.layers.Embedding(len(all_ages), small_embedding_size, name='age_emb')(self.age_sl), axis=1)
        
        self.occupation_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='occupation_input')
        self.occupation_sl = tf.keras.layers.StringLookup(vocabulary=all_occupations, num_oov_indices=0, name='occupation_string_lookup')(self.occupation_input)
        self.occupation_emb = tf.squeeze(tf.keras.layers.Embedding(len(all_occupations), small_embedding_size, name='occupation_emb')(self.occupation_sl), axis=1)
        
        self.gender_input = tf.keras.Input(shape=(1,), name='gender_input')
        self.hour_input = tf.keras.Input(shape=(1,), name='hour_input')
        self.day_input = tf.keras.Input(shape=(1,), name='day_input')
        self.month_input = tf.keras.Input(shape=(1,), name='month_input')
        
        self.user_merged = tf.keras.layers.concatenate([self.user_emb, self.city_emb, self.state_emb, self.age_emb, 
                                                        self.occupation_emb, self.gender_input, self.hour_input,
                                                        self.day_input, self.month_input], 
                                                       axis=-1, name='user_merged')
        self.user_dense = tf.keras.layers.Dense(100, activation='relu', name='user_dense')(self.user_merged)
        self.user_last_layer = tf.keras.layers.Dense(tower_last_layer_size, activation='relu', name='user_last_layer')(self.user_dense)
        
        # Movie tower
        
        self.movie_input = tf.keras.Input(shape=(1,), dtype=tf.string, name='movie_input')
        self.movie_sl = tf.keras.layers.StringLookup(vocabulary=all_movies, name='movie_string_lookup')(self.movie_input)
        self.movie_emb = tf.squeeze(tf.keras.layers.Embedding(len(all_movies)+1, large_embedding_size, name='movie_emb')(self.movie_sl), axis=1)
        
        self.title_input = tf.keras.Input(shape=(title_emb_len,), name='title_input')
        self.title_dense = tf.keras.layers.Dense(title_emb_len, activation='softmax', name='title_softmax')(self.title_input)
        
        self.genres_input = tf.keras.Input(shape=(len(all_genres),), name='genres_input')
        self.year_input = tf.keras.Input(shape=(1,), name='year_input')
        
        self.movie_merged = tf.keras.layers.concatenate([self.movie_emb, self.title_dense, self.genres_input, self.year_input] ,axis=-1, name='movie_merged')
        self.movie_dense = tf.keras.layers.Dense(100, activation='relu', name='movie_dense')(self.movie_merged)
        self.movie_last_layer = tf.keras.layers.Dense(tower_last_layer_size, activation='relu', name='movie_last_layer')(self.movie_dense)
        
        # Combining towers
        
        self.towers_multiplied = tf.keras.layers.Multiply(name='towers_multiplied')([self.user_last_layer, self.movie_last_layer])
        self.towers_dense1 = tf.keras.layers.Dense(40, activation='relu', name='towers_dense1')(self.towers_multiplied)
        self.towers_dense2 = tf.keras.layers.Dense(20, activation='relu', name='towers_dense2')(self.towers_dense1)
        self.output_node = tf.keras.layers.Dense(1, name='output_node')(self.towers_dense2)
        
        # Model definition
        
        self.model = tf.keras.Model(inputs={'userId': self.user_input, 
                                            'city': self.city_input,
                                            'state': self.state_input,
                                            'age': self.age_input,
                                            'occupation': self.occupation_input,
                                            'gender': self.gender_input,
                                            'hour': self.hour_input,
                                            'day': self.day_input,
                                            'month': self.month_input,
                                            'movieId': self.movie_input,
                                            'title': self.title_input,
                                            'genres': self.genres_input,
                                            'year': self.year_input
                                            }, 
                                    outputs=self.output_node)
        
        self.task = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )
        
    def call(self, features):
        return self.model({'userId': tf.strings.as_string(features["userId"]), 
                           'city': features["city"], 
                           'state': features["state"],
                           'age': tf.strings.as_string(features["age"]),
                           'occupation': tf.strings.as_string(features["occupation"]), 
                           'gender': features["gender"],
                           'hour': features["hour"],
                           'day': features["day"],
                           'month': features["month"],
                           'movieId': tf.strings.as_string(features["movieId"]),
                           'title': features["title_emb"],
                           'genres': features["genres"],
                           'year': features["movie_year"]
                           })
    
    def compute_loss(self, features, **kwargs):
        return self.task(labels=features["rating"], predictions=self(features))


loaded_model = RatingPredictionModel()
learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(2e-3, decay_steps=4000, decay_rate=0.95)
loaded_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))


loaded_model.built = True
loaded_model.load_weights("model_val_weights.h5")

def genre_as_feature_name(genre): 
    # Here we remove any non-alphanumeric characters from the genre's name, and add a 'genre_' suffix to it
    return 'genre_'+re.sub('[\W_]+', '', genre.lower())


def convert_to_dataset(df):
    d = {k:v.to_numpy() for k,v in dict(df).items()}
    d['genres'] = np.transpose(np.array([d[x] for x in all_genres]))
    d['title_emb'] = np.transpose(np.array([d[f'title_emb_{i}'] for i in range(title_emb_len)]))
    for x in all_genres + [f'title_emb_{i}' for i in range(title_emb_len)]:
        d.pop(x)
    return tf.data.Dataset.from_tensor_slices(d)


def make_dataset(user_ids_batch, user_df, movies_df, epoch_time = 1680480000, num_recs = 15):
    selected_user_data = user_df[user_df['userId'].isin(user_ids_batch)][['userId', 'state', 'city', 'age', 'occupation', 'gender']].drop_duplicates()
    user_metadata_batch = pd.concat([selected_user_data] * len(movies_df), ignore_index=True).sort_values(by='userId').reset_index(drop=True)
    # Duplicate top_genres_df for each user in the batch
    top_genres_batch = pd.concat([movies_df] * len(user_ids_batch), ignore_index=True)

    # Combine user metadata and top genres DataFrames
    combined_df = pd.concat([user_metadata_batch, top_genres_batch], axis=1)
    combined_df['hour'] = datetime.datetime.fromtimestamp(epoch_time).hour
    combined_df['day'] = datetime.datetime.fromtimestamp(epoch_time).day
    combined_df['month'] = datetime.datetime.fromtimestamp(epoch_time).month
        
    all_genres = combined_df['genres'].explode().dropna().unique()
    combined_df[[genre_as_feature_name(g) for g in all_genres]] = combined_df.apply(lambda row: pd.Series({genre_as_feature_name(g): float(g in row['genres']) for g in all_genres}), axis=1)
    combined_df = combined_df.drop('genres', axis=1)
    
    combined_df[[f'title_emb_{i}' for i in range(25)]] = combined_df.apply(embbed_title, axis=1)
    combined_df = combined_df.drop('title', axis=1)
    combined_df = combined_df.astype({'movie_year': 'int64'})

    return combined_df 


def safe_embbed_word(word, length=25):
    try:
        return word2vec[word.lower()]
    except KeyError:
        return np.zeros(length)
    
def embbed_title(row):
    words = strip_multiple_whitespaces(remove_stopwords(split_alphanum(strip_non_alphanum(strip_punctuation(row['title']))))).split(' ')  # all functions are imported from gensim.parsing.preprocessing
    embs = np.array([safe_embbed_word(w) for w in words])
    return pd.Series({f'title_emb_{i}': v for i,v in enumerate(embs.sum(axis=0))})

def get_recommendation(user_id, top=10):
    top10 = []
    print(user_id)
    # users = pd.read_csv("data/users.csv")
    # movies = pd.read_csv("data/movies.csv")
    final_test_data = make_dataset([user_id], users, movies)
    print(final_test_data)
    final_test_data = final_test_data.drop_duplicates().dropna()
    for name, group in final_test_data.groupby("userId"):
        test_df = convert_to_dataset(group)
        cached_test = test_df.batch(500).cache()
        preds = loaded_model.predict(cached_test)
        group["predicted_rating"] = preds
        print('$')
        temp = group.sort_values(by='predicted_rating', axis=0, ascending=False).reset_index(drop=True)
        for movieid, rating in zip(temp.head(top)["movieId"].values.tolist(), temp.head(top)["predicted_rating"].values.tolist()):
            print('#')
            top10.append(movies.loc[movies['movieId'] == movieid]["title"].values[0])
        return top10
