import enum
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple, defaultdict

from clip import tokenize

instance_fields = [
    'text',
    'text_vec',
]
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))

batch_fields = [
    'text',
    'text_vec',
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


class TextDataset(Dataset):
    def __init__(self, text_list, tokenize=None, device='gpu'):
        """
        :param path (str): path to the data file.
        :param gpu (bool): use GPU (default=False).
        """
        self.text_list = text_list
        self.data = []

        self.device = device
        self.tokenize = tokenize

        self.load_data()

    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        return self.data[item]


    def load_data(self):
        """Load data from file."""
        for text in self.text_list:
            inst = dict()
            inst['text'] = text
            self.data.append(inst)
        
        # print('Loaded {} text instances'.format(len(self)))

    def collate_fn(self, batch): #, preprocess, tokenize):
        
        text_list = list()
        text_vecs = None
        
        for inst in batch:
            text_list.append(inst['text'])
            text_vec = self.tokenize(inst['text']).to(self.device) #.unsqueeze(0)
            text_vecs = torch.cat((text_vecs, text_vec), dim=0) if text_vecs is not None else text_vec

        return Batch(
            text=text_list,
            text_vec=text_vecs
        )


if __name__ == '__main__':
    text_list = ['test1', 'test2', 'test3']
    dataset = TextDataset(
        text_list = text_list,
        tokenize=tokenize, device=torch.device('cuda')
    )
    loader = DataLoader(
        dataset, batch_size=2, 
        collate_fn=dataset.collate_fn,
        pin_memory=False
    )

    for batch_idx, batch in enumerate(loader):
        # print(batch)
        text = batch.text
        text_vec = batch.text_vec
        
        print(text)
        print(text_vec)

        break
        