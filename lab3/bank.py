import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix

bank = pd.read_csv("bank-additional-full.csv")
print(bank.shape)



data = pd.get_dummies(bank, columns=["job", "marital", "education", "default", "housing", "loan", "contact",
                                           "month", "day_of_week", "campaign", "pdays", "previous", "poutcome", "y"])

# data = pd.get_dummies(bank)
data.pop("y_no")
data.pop("duration")
sns.distplot(data["y_yes"])

train_data = data.drop(columns=["y_yes"])
train_target = data[["y_yes"]]

# Maximum number of levels in tree
max_depth = [5, 10, 15, None]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5]

grid = {
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        }


#clf = GridSearchCV(estimator=RandomForestClassifier(n_estimators=2000), param_grid=grid, n_jobs=-1)

score = []
k_fold = KFold(n_splits=5)
"""
for train_indices, test_indices in k_fold.split(train_data):
    clf.fit(train_data, train_target)
    score.append(clf.score(train_data[test_indices], train_target[test_indices]))
    print('Best Max Depth:', clf.best_estimator_.max_depth,
          'Best min samples split:', clf.best_estimator_.min_samples_split,
          'Best min samples leaf:', clf.best_estimator_.min_samples_leaf,
          'Fold test accuracy:', score[-1])
print('Average accuracy: {} %'.format(np.mean(score) * 100))

"""
clf = RandomForestClassifier(n_estimators=2000)
score_tree = cross_val_score(clf, train_data, train_target, cv=k_fold, n_jobs=-1)
print('Average accuracy:', np.mean(score_tree))

clf.fit(train_data, train_target)

# Plot non-normalized confusion matrix
predict = clf.predict(train_data)
cm = confusion_matrix(train_target, predict)
best_features = clf.feature_importances_

importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
print(indices)
# Print the feature ranking
print("Feature ranking:")

for f in range(len(data.columns)):
    print("%d. %s (%f)" % (f + 1, data[indices[f]],  importances[indices[f]]))
print("DONE")