import enum
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple, defaultdict

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

from torchvision.datasets.folder import DatasetFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip import tokenize

instance_fields = [
    'image_id',
    'image_vec',
    'captions',
    'captions_vec'
]
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))

batch_fields = [
    'image_id',
    'image_vec',
    'captions',
    'captions_vec'
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


class FlickerDataset(Dataset):
    def __init__(self, split_list, caption_file, image_dir, preprocess=None, tokenize=None, device='gpu'):
        """
        :param path (str): path to the data file.
        :param gpu (bool): use GPU (default=False).
        """
        self.split_list = split_list
        self.caption_file = caption_file
        self.image_dir = image_dir
        self.data = []

        self.device = device
        self.preprocess = preprocess
        self.tokenize = tokenize

        self.load_data()

    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        # image reading

        return self.data[item]


    def load_data(self):
        """Load data from file."""

        # load caption pairs
        caption_dict = defaultdict(list) #defaultdict(lambda : defaultdict())
        for line in open(self.caption_file):
            line = line.rstrip('\n')
            tabs = line.split('|')
            image_id = tabs[0].strip()
            caption_idx = tabs[1].strip()
            # print(line)
            caption = 'An photo of ' + tabs[2].strip()
            # caption = 'An image of %s' % tabs[2].strip()
            # caption = 'This is an image of %s' % tabs[2].strip()
            # caption = tabs[2].strip()
            # caption_dict[image_id][caption_idx] = caption
            caption_dict[image_id].append(caption)
        
        # load images
        for line in open(self.split_list):
            image_id = line.rstrip('\n')
            image_id = image_id+'.jpg'
            inst = dict()
            inst['image_id'] = image_id
            if image_id not in caption_dict:
                print('no captions %s .' % image_id)
            inst['captions'] = caption_dict[image_id]
            assert len(inst['captions']) == 5
            
            self.data.append(inst)

            # break
        
        print('Loaded {} instances from {}'.format(len(self), self.split_list))

    def collate_fn(self, batch): #, preprocess, tokenize):
        
        image_ids = list()
        image_vecs = list()
        captions = list() # each image has five captions
        captions_vecs = list()
        
        for inst in batch:
            image_ids.append(inst['image_id'])

            image_path = os.path.join(self.image_dir, inst['image_id'])
            image_obj = Image.open(image_path)
            image_vec = self.preprocess(image_obj).to(self.device)
            image_vecs.append(image_vec)

            captions.append(inst['captions'])  # 5 captions
            captions_vecs.append(self.tokenize(inst['captions']).to(self.device))

                
        image_vecs = torch.stack(image_vecs, dim=0).to(self.device)
        # argbbox_vecs = torch.stack(argbbox_vecs, dim=0).to(self.device)
        captions_vecs = torch.stack(captions_vecs, dim=0).to(self.device)#.squeeze(1)
        # desc_argrole_vecs = torch.stack(desc_argrole_vecs, dim=0).to(self.device)

        return Batch(
            image_id=image_ids,
            image_vec=image_vecs,
            captions=captions,
            captions_vec=captions_vecs
        )


if __name__ == '__main__':
    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    dataset = FlickerDataset(
        split_list='/shared/nas/data/m1/manling2/clip-event/data/flicker30k/train.txt', 
        caption_file='/shared/nas/data/m1/manling2/clip-event/data/flicker30k/flicker30k_captions.csv', 
        image_dir='/shared/nas/data/m1/manling2/clip-event/data/flicker30k/flickr30k-images', 
        preprocess=transform, tokenize=tokenize, device=torch.device('cuda')
    )
    loader = DataLoader(
        dataset, batch_size=2, 
        collate_fn=dataset.collate_fn,
        pin_memory=False
    )

    for batch_idx, batch in enumerate(loader):
        # print(batch)
        image_id = batch.image_id
        image_vec = batch.image_vec
        captions = batch.captions
        captions_vec = batch.captions_vec

        # print(image_id)

        # print(image_vec.size(), captions_vec.size())  # torch.Size([2, 3, 224, 224]) torch.Size([2, 5, 77])

        # break
        if len(image_id) != 200:
            print(image_id)