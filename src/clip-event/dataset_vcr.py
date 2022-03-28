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

instance_fields = [
    'anno_id',
    'image_vec',
    'description_vec',
    'labels_per_image'
]
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))

batch_fields = [
    'anno_id',
    'image_vec',
    'description_vec',
    'labels_per_image'
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


class VCRDataset(Dataset):
    def __init__(self, qa_json, image_dir, retionale=False, preprocess=None, tokenize=None, device='gpu'):
        """
        :param path (str): path to the data file.
        :param gpu (bool): use GPU (default=False).
        """
        self.qa_json = qa_json
        self.image_dir = image_dir
        self.retionale = retionale
        self.data = []

        self.device = device
        self.preprocess = preprocess
        self.tokenize = tokenize

        # self.vocabs, self.templates = self.load_dict_evt()
        self.load_data()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        # image reading

        return self.data[item]

    def shorten_context(self, text):
        text = text.replace('FILE - ', '')
        text = text[:350] # not exceed 350 characters
        return text


    def load_data(self):
        """Load data from file."""

        # load qa pairs
        for line in open(self.qa_json):
            data = json.loads(line)
            anno_id = data['annot_id']
            movie = data['movie']
            objects = data['objects']
            # image path
            img_fn = data['img_fn']
            # bounding boxes
            metadata_fn = data['metadata_fn']
            question = data ['question']
            answer_choices = data['answer_choices']
            answer_label = data['answer_label']
            rationale_choices = data['rationale_choices']
            rationale_label = data['rationale_label']

            inst = dict()
            inst['anno_id'] = anno_id
            # inst['movie'] = movie
            inst['image'] = img_fn
            # inst['answer'] = list()
            # inst['retionale'] = list()
            inst['descriptions'] = list()

            question_str = self.fill_name(question, objects)
            inst['question'] = question_str

            if self.retionale:
                for retionale in rationale_choices:
                    retionale_str = self.fill_name(retionale, objects)
                    inst['descriptions'].append(retionale_str)
                    inst['label'] = rationale_label
            else:
                for answer in answer_choices:
                    answer_str = self.fill_name(answer, objects)
                    inst['descriptions'].append(answer_str)
                    inst['label'] = answer_label
            
            self.data.append(inst)

            # break
        
        print('Loaded {} instances from {}'.format(len(self), self.qa_json))

    def fill_name(self, word_list, object_names):
        for word_idx, word in enumerate(word_list):
            if isinstance(word, list):
                word_objnames = [object_names[obj_idx] for obj_idx in word]
                word_list[word_idx] = ' and '.join(word_objnames)
        return ' '.join(word_list)


    def clean_imageid(self, image_id):
        return image_id.replace('.', '_')

    def collate_fn(self, batch): #, preprocess, tokenize):
        # print('batch', batch[0]['image_id'])    

        # image_ids = [self.clean_imageid(inst['image_id']) for inst in batch]
        anno_ids = list()
        # movies = list()
        images = list()
        image_vecs = list()
        description = list()
        description_vecs = list()
        
        for inst in batch:
            anno_ids.append(inst['anno_id'])
            # movies.append(inst['movie'])
            images.append(inst['image'])
            description_vecs.append(self.tokenize(inst['descriptions']).to(self.device))

            image_path = os.path.join(self.image_dir, inst['image'])
            image_vec = self.preprocess(Image.open(image_path)).to(self.device)
            image_vecs.append(image_vec)
        image_vecs = torch.stack(image_vecs, dim=0).to(self.device)
        
        description_vecs = torch.stack(description_vecs, dim=0)
        description_vecs = description_vecs.view(len(batch)*4, -1)

        labels_per_image_keepshape = [inst['label'] for inst in batch]
        labels_per_image_keepshape = torch.LongTensor(labels_per_image_keepshape).to(self.device)

        return Batch(
            anno_id=anno_ids,
            image_vec=image_vecs,
            description_vec=description_vecs,
            labels_per_image=labels_per_image_keepshape
        )


if __name__ == '__main__':
    dataset = VCRDataset(
        qa_json='/shared/nas/data/m1/manling2/clip-event/data/vcr/val.jsonl', 
        image_dir='/shared/nas/data/m1/manling2/clip-event/data/vcr/images/vcr1images', 
        retionale=False,
        device='cuda'
    )
    loader = DataLoader(
        dataset, batch_size=2, 
        collate_fn=dataset.collate_fn,
        pin_memory=False
    )

    for batch_idx, batch in enumerate(loader):
        # print(batch)
        anno_id = batch.anno_id
        image_vec = batch.image_vec
        description_vec = batch.description_vec

        print(anno_id)