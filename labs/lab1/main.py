import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("abalone.csv")

V1 = {'F': 0, 'I': 1, 'M': 2}
dataset['V1'] = dataset['V1'].map(V1)

columns = list(dataset.columns[:-1])
for column in columns[1:]:
    dataset[column] = pd.qcut(dataset[column], 4, labels=False)


def countplot(x_name, dataframe, sub):
    plt.subplot(sub)
    sns.countplot(x=x_name, hue='Class', data=dataframe)


plt.figure(figsize=(16, 4))
sub = 241

for column in columns:
    print(column, sub)
    countplot(column, dataset, sub)
    sub += 1

plt.subplots_adjust(hspace=0.5)
# plt.show()


X = dataset[columns]
y = dataset['Class']

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2020)
knn = KNeighborsRegressor(n_neighbors=3, metric='minkowski', p=1)
knn.fit(X_train, y_train)
y_pred = np.round(knn.predict(X_test))

print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='weighted'))

param_grid = {
    'n_neighbors': [1, 3, 5, 10, 50, 100],
    'metric': ['minkowski', 'euclidean', 'chebyshev'],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 3, 4]
}

knn_boosting = KNeighborsRegressor()
clf = GridSearchCV(estimator=knn_boosting, param_grid=param_grid, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.best_params_)

knn = KNeighborsRegressor(**clf.best_params_)
knn.fit(X_train, y_train)
y_pred = np.round(knn.predict(X_test))
print(confusion_matrix(y_test, y_pred))
print(f1_score(y_test, y_pred, average='weighted'))


