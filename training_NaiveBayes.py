import json
from nltk_utils import token, stem, bag_of_words
import numpy as np
from sklearn.naive_bayes import BernoulliNB

with open('content.json', 'r') as c:
    contents = json.load(c)
with open('test_content.json', 'r') as t:
    test_contents = json.load(t)

all_words = []
tags = []
xy = []
xy_test = []
punctuation = ['?', '.', ',', '!', ':', '/']
for content in contents['intents']:
    tag = content['tag']
    tags.append(tag)
    for pattern in content['patterns']:
        w = token(pattern)
        all_words.extend(w)
        xy.append((w, tag))
for test_content in test_contents['intents']:
    tag = test_content['tag']
    for test_pattern in test_content['patterns']:
        t_w = token(test_pattern)
        xy_test.append((t_w, tag))

all_words = sorted(set([stem(w) for w in all_words if w not in punctuation]))
tags = sorted(set(tags))
print(tags)

X_train = []
y_train = []
X_test = []
y_test = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    label = tags.index(tag)
    X_train.append(bag)
    y_train.append(label)
for (pattern_test, tag) in xy_test:
    bag = bag_of_words(pattern_test, all_words)
    label = tags.index(tag)
    X_test.append(bag)
    y_test.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Model:
model = BernoulliNB()

# Training:
model.fit(X_train, y_train)

# Testing:
prediction = model.predict(X_test)
print(prediction)
print(y_test)
correct = 0
for i in range(len(prediction)):
    if prediction[i] == y_test[i]:
        correct += 1
print(correct/len(prediction))
