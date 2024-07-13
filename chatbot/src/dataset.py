import os
import torch
from torch import nn
import json
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import AutoTokenizer, AdamW, AutoModel
from vncorenlp import VnCoreNLP
# from py_vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP('/home/trunglx/Downloads/github/my_chatbot/chatbot/vncorenlp/VnCoreNLP-1.1.1.jar', annotators="wseg", max_heap_size='-Xmx2g')
# rdrsegmenter = VnCoreNLP(address="http://127.0.0.1", port=9000)
# rdrsegmenter = VnCoreNLP(annotators="wseg", save_dir='chatbot/vncorenlp')
BATCH_SIZE = 4


class ChatbotDataset(Dataset):
    def __init__(self, data_dir, tokenizer, batch_size):
        super(ChatbotDataset, self).__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        with open(self.data_dir, 'r', encoding='utf-8') as f:
            contents = json.load(f)
        tags = []
        X = []
        y = []
        for content in contents['intents']:
            tag = content['tag']
            for pattern in content['patterns']:
                X.append(pattern)
                tags.append(tag)
        tag_set = sorted(set(tags))
        for tag in tags:
            label = tag_set.index(tag)
            y.append(label)

        encode_dict = self.tokenizer.encode_plus(X,
                                                 max_length=64,
                                                 padding='max_length',
                                                 truncation=True,
                                                 return_attention_mask=True,
                                                 return_token_type_ids=False,
                                                 return_tensors='pt'
                                                 )
        input_ids = encode_dict['input_ids'][0]
        attention_mask = encode_dict['attention_mask'][0]

        y_train = torch.LongTensor(y)
        return input_ids, attention_mask, y_train

def data_loader(data_dir, tokenizer, batch_size=BATCH_SIZE):
    with open(data_dir, 'r', encoding='utf-8') as f:
        contents = json.load(f)

    tags = []
    X = []
    y = []
    for content in contents['intents']:
        tag = content['tag']
        for pattern in content['patterns']:
            X.append(pattern)
            tags.append(tag)

    tag_set = sorted(set(tags))
    for tag in tags:
        label = tag_set.index(tag)
        y.append(label)

    X = [' '.join(rdrsegmenter.tokenize(x)[0]) for x in X]
    token_train = {}
    token_train = tokenizer.batch_encode_plus(X,
                                              max_length=64,
                                              padding='max_length',
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_token_type_ids=False,
                                              return_tensors='pt'
                                              )

    X_train_mask = token_train['attention_mask']
    X_train = token_train['input_ids']
    y_train = torch.LongTensor(y)

    dataset = TensorDataset(X_train, X_train_mask, y_train)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    return data_loader

if __name__ == '__main__':
    train_dir = '/home/trunglx/Downloads/github/my_chatbot/chatbot/data/intent_train.json'
    val_dir = '/home/trunglx/Downloads/github/my_chatbot/chatbot/data/intent_val.json'
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
    train_data_loader = data_loader(train_dir, tokenizer, batch_size=BATCH_SIZE)
    val_data_loader = data_loader(val_dir, tokenizer, batch_size=BATCH_SIZE)
    print(train_data_loader)
