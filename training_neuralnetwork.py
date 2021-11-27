import json
from model import MyNeuralNet
from nltk_utils import token, stem, bag_of_words
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


with open('content.json', 'r') as c:
    contents = json.load(c)

all_words = []
tags = []
xy = []
punctuation = ['?', '.', ',', '!', ':', '/']

for content in contents['intents']:
    tag = content['tag']
    tags.append(tag)
    for pattern in content['patterns']:
        w = token(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = sorted(set([stem(w) for w in all_words if w not in punctuation]))
tags = sorted(set(tags))

X_train = []
y_train = []
X_test = []
y_test = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    label = tags.index(tag)
    X_train.append(bag)
    y_train.append(label)   # CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)


class Chatdata(Dataset):
    def __init__(self):
        self.n_sample = len(X_train)
        self.X_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.n_sample


# Hyperparameters:
batch_size = 8
input_size = len(all_words)
hidden_size = 7
num_class = len(tags)
learning_rate = 1e-3
num_epoch = 1000

# Data:
dataset = Chatdata()
train_data = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Model:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = MyNeuralNet(input_size, hidden_size, num_class)

# Loss and optimizer:
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epoch):
    for (words, label) in train_data:
        words = words.to(device)
        label = label.to(dtype=torch.long).to(device)
        output = model(words)
        loss = loss_f(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1} -------------')
        print(f'    Losses: {loss}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": num_class,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
