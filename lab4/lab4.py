import pandas as pd
import numpy as np
import random
from IPython.display import Image
import operator

jester = pd.read_csv("jester-data-1.csv")
jester_train = jester
print(jester.shape)

num_values = jester.shape[0] * jester.shape[1]
print(num_values)

num_99 = 0
for column in range(0, jester.shape[0]):
    for row in range(0, jester.shape[1]):
        value = jester.iat[column, row]
        if value == 99:
            num_99 += 1

print("Number 99 is ", num_99)

num_changes = 0
validation = num_values / 10
print(validation)
while num_changes < validation:
    random_row = random.randint(0, jester.shape[1] - 1)
    random_column = random.randint(0, jester.shape[0] - 1)
    value = jester_train.iat[random_column, random_row]
    if value == 99:
        continue
    jester_train.iat[random_column, random_row] = 99
    num_changes += 1

n_features = 3

latent_user_preferences = np.random.random((jester_train.shape[0], n_features))
latent_joke_features = np.random.random((jester_train.shape[1], n_features))


def predict_rating(user_id, joke_id):
    """ Predict a rating given a user and joke
    """
    user_preference = latent_user_preferences[user_id]
    item_preference = latent_joke_features[joke_id]
    return user_preference.dot(item_preference)


def train(user_id, joke_id, rating, alpha=0.0001):
    # print joke_id
    prediction_rating = predict_rating(user_id, joke_id)
    err = (prediction_rating - rating)
    # print err
    user_pref_values = latent_user_preferences[user_id][:]
    latent_user_preferences[user_id] -= alpha * err * latent_joke_features[joke_id]
    latent_joke_features[joke_id] -= alpha * err * user_pref_values
    return err

def test(user_id, joke_id,rating):
    print()
    prediction_rating = predict_rating(user_id, joke_id)
    err = (prediction_rating - rating)
    return err

def sgd(iterations=100):
    """ Iterate over all users and all items and train for
        a certain number of iterations
    """
    for iteration in range(0, iterations):
        error = []
        for user_id in range(0, latent_user_preferences.shape[0]):
            for joke_id in range(0, latent_joke_features.shape[0]):
                rating = jester_train.iat[user_id, joke_id]
                if rating != 99:  # not null
                    err = train(user_id, joke_id, rating)
                    error.append(err)
        mse = (np.array(error) ** 2).mean()
        print(iteration, mse)

"""
def validation():
    error = []
    for user_id in range(0, latent_user_preferences.shape[0]):
        for item_id in range(0, latent_joke_features.shape[0]):
            rating = jester_train.iat[user_id, item_id]
            if rating == 99 and jester.iat[user_id,item_id]!= 99:  # only null in train data
                err = test(user_id, item_id, rating)
                error.append(err)
    mse = (np.array(error) ** 2).mean()
    print("Validation MSE ",mse)
"""

sgd()
validation()

"""
joke_scores = []
for column in jester:
    joke_scores.append((jester[column].mean()))

min_index, min_value = min(enumerate(joke_scores), key=operator.itemgetter(1))
max_index, max_value = max(enumerate(joke_scores), key=operator.itemgetter(1))
print("Best joke is joke {} with a score of {}".format(jester.columns.values[max_index], max_value))
print("Worst joke is joke {} with a score of {}".format(jester.columns.values[min_index], min_value))
"""
