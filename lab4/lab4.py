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
validation = num_values/10
print(validation)
while num_changes < validation:
    random_row = random.randint(0,jester.shape[1]-1)
    random_column = random.randint(0,jester.shape[0]-1)
    value = jester_train.iat[random_column,random_row]
    if value == 99:
        continue
    jester_train.iat[random_column, random_row] = 99
    num_changes += 1


"""
joke_scores = []
for column in jester:
    joke_scores.append((jester[column].mean()))

min_index, min_value = min(enumerate(joke_scores), key=operator.itemgetter(1))
max_index, max_value = max(enumerate(joke_scores), key=operator.itemgetter(1))
print("Best joke is joke {} with a score of {}".format(jester.columns.values[max_index], max_value))
print("Worst joke is joke {} with a score of {}".format(jester.columns.values[min_index], min_value))
"""
