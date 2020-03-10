import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from bayes import NaiveBayesClassifier

messages_dir = './messages/part'
messages = []

for i in range(1, 11):
    messages_part = []
    for message_filename in os.listdir(messages_dir + str(i)):
        message_file = open(messages_dir + str(i) + "/" + message_filename, "r")
        subject = message_file.readline()[9:-1]
        message_file.readline()
        text = message_file.readline()[:-1]
        is_spam = 1 if 'spmsg' in message_filename else 0

        messages.append([subject, text, is_spam])

messages = pd.DataFrame(messages, columns=['subject', 'text', 'spam'])
print(messages)

X = messages['text']
y = messages['spam']

vectorizer = CountVectorizer(ngram_range=(1, 1))
X = vectorizer.fit_transform(X).toarray()
# print(vectorizer.get_feature_names()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

spam_classifier = NaiveBayesClassifier()
spam_classifier.fit(X_train, y_train)
y_pred = spam_classifier.predict(X_train)

print(accuracy_score(y_train, y_pred))
print(f1_score(y_train, y_pred))
print(confusion_matrix(y_train, y_pred))

y_pred = spam_classifier.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
