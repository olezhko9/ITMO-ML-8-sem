import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes import NaiveBayesClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_curve
from sklearn.model_selection import cross_validate

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

spam_classifier = NaiveBayesClassifier()
spam_classifier.fit(X, y)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1_score': make_scorer(f1_score)
}
score = cross_validate(spam_classifier, X, y, cv=10, scoring=scoring)

print('accuracy:', np.mean(score['test_accuracy']))
print('f1_score:', np.mean(score['test_f1_score']))

y_pred_proba = spam_classifier.predict_proba(X)


def plot_roc_curve(y, y_proba):
    fpr, tpr, thresholds = roc_curve(y, y_proba[:, 1])
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


plot_roc_curve(y, y_pred_proba)
