import json
import random
import torch
from model import MyNeuralNet
from nltk_utils import bag_of_words, token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('content.json', 'r') as json_data:
    contents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
num_class = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = MyNeuralNet(input_size, hidden_size, num_class)
model.load_state_dict(model_state)
model.eval()


def chat_bot(sentence):
    sentence = token(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = model(X)
    _, predict = torch.max(output, dim=1)
    tag = tags[predict.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predict.item()]
    if prob.item() > 0.75:
        for content in contents['intents']:
            if tag == content['tag']:
                answer = random.choice(content['responses'])
    else:
        answer = "I don't understand you, please be more clear!"
    return answer
