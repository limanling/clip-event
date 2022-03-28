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


class COCODataset(Dataset):
    def __init__(self, caption_file, image_dir, prompt='An photo of', preprocess=None, tokenize=None, device='gpu'):
        """
        :param path (str): path to the data file.
        :param gpu (bool): use GPU (default=False).
        """
        # self.split_list = split_list
        self.caption_file = caption_file
        self.image_dir = image_dir
        self.data = []

        self.device = device
        self.preprocess = preprocess
        self.tokenize = tokenize
        self.prompt = prompt

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
        data_all = json.load(open(self.caption_file))
        # {'filepath': 'val2014', 'sentids': [770337, 771687, 772707, 776154, 781998], 'filename': 'COCO_val2014_000000391895.jpg', 'imgid': 0, 'split': 'test', 'sentences': [{'tokens': ['a', 'man', 'with', 'a', 'red', 'helmet', 'on', 'a', 'small', 'moped', 'on', 'a', 'dirt', 'road'], 'raw': 'A man with a red helmet on a small moped on a dirt road. ', 'imgid': 0, 'sentid': 770337}, {'tokens': ['man', 'riding', 'a', 'motor', 'bike', 'on', 'a', 'dirt', 'road', 'on', 'the', 'countryside'], 'raw': 'Man riding a motor bike on a dirt road on the countryside.', 'imgid': 0, 'sentid': 771687}, {'tokens': ['a', 'man', 'riding', 'on', 'the', 'back', 'of', 'a', 'motorcycle'], 'raw': 'A man riding on the back of a motorcycle.', 'imgid': 0, 'sentid': 772707}, {'tokens': ['a', 'dirt', 'path', 'with', 'a', 'young', 'person', 'on', 'a', 'motor', 'bike', 'rests', 'to', 'the', 'foreground', 'of', 'a', 'verdant', 'area', 'with', 'a', 'bridge', 'and', 'a', 'background', 'of', 'cloud', 'wreathed', 'mountains'], 'raw': 'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', 'imgid': 0, 'sentid': 776154}, {'tokens': ['a', 'man', 'in', 'a', 'red', 'shirt', 'and', 'a', 'red', 'hat', 'is', 'on', 'a', 'motorcycle', 'on', 'a', 'hill', 'side'], 'raw': 'A man in a red shirt and a red hat is on a motorcycle on a hill side.', 'imgid': 0, 'sentid': 781998}], 'cocoid': 391895}
        for data in data_all['images']:
            image_id = data['filename'].split('_')[-1]
            # print(filename)
            for sentence_data in data['sentences']:
                sentence = sentence_data['raw']
                caption = self.prompt + sentence.lower() #'An photo of ' + sentence.lower()
                caption_dict[image_id].append(caption)            
        
        # load images
        for image_id in os.listdir(self.image_dir):
            inst = dict()
            inst['image_id'] = image_id
            if image_id not in caption_dict:
                raise RuntimeError("No captions '{}'. ".format(image_id))
            inst['captions'] = caption_dict[image_id]
            if len(inst['captions']) > 5:
                print(image_id, len(inst['captions']))
                # 000000165257.jpg 6
                # 000000431896.jpg 7
                # 000000449312.jpg 6
                # 000000096493.jpg 6
                # 000000163057.jpg 6
                # 000000328030.jpg 6
                # 000000289516.jpg 6
                # 000000215259.jpg 6
                # 000000002923.jpg 6
                # 000000545958.jpg 6
                # 000000038070.jpg 6
                # 000000190841.jpg 6
                # 000000434459.jpg 6
            
            self.data.append(inst)

            # break
        
        print('Loaded {} instances from {}'.format(len(self), self.image_dir))

    def collate_fn(self, batch): #, preprocess, tokenize):
        
        image_ids = list()
        image_vecs = None
        captions = list() # each image has five captions
        captions_vecs = None
        
        for inst in batch:
            image_ids.append(inst['image_id'])

            image_path = os.path.join(self.image_dir, inst['image_id'])
            image_obj = Image.open(image_path)
            image_vec = self.preprocess(image_obj).to(self.device).unsqueeze(0)
            image_vecs = torch.cat((image_vecs, image_vec), dim=0) if image_vecs is not None else image_vec

            captions.append(inst['captions'][:5])  # 5 captions
            captions_vec = self.tokenize(inst['captions'][:5]).to(self.device).unsqueeze(0)
            # print(inst['image_id'], len(inst['captions']), 'captions_vec', captions_vec.size())
            captions_vecs = torch.cat((captions_vecs, captions_vec), dim=0) if captions_vecs is not None else captions_vec
            # print('captions_vecs', captions_vecs.size(), 'captions_vec', captions_vec.size())

            # assert captions_vec.size(0) == 

        # argbbox_vecs = torch.stack(argbbox_vecs, dim=0).to(self.device)
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

    dataset = COCODataset(
        # split_list='/shared/nas/data/m1/manling2/clip-event/data/flicker30k/train.txt', 
        caption_file='/shared/nas/data/m1/manling2/clip-event/data/mscoco/caption_datasets/dataset_coco.json', 
        image_dir='/shared/nas/data/m1/manling2/multimedia-common-space/data/coco/2017/val2017', 
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

        print(image_id, captions)

        print(image_vec.size(), captions_vec.size())  # torch.Size([2, 3, 224, 224]) torch.Size([2, 5, 77])

        break
        