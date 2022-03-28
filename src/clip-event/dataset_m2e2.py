import os
import copy
import itertools
import json
import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict

import clip
from PIL import Image
import requests

instance_fields = [
    'image_id',
    'image_vec',
    'candidates_vec',
    'event_type_idx'
    # 'sent_id', 'tokens', 'pieces', 'piece_idxs', 'token_lens', 'attention_mask',
    # 'entity_label_idxs', 'trigger_label_idxs',
    # 'entity_type_idxs', 'event_type_idxs',
    # 'relation_type_idxs', 'role_type_idxs',
    # 'mention_type_idxs',
    # 'graph', 'entity_num', 'trigger_num'
]
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))

batch_fields = [
    'image_id',
    'image_vec',
    'candidates_vec',
    'event_type_idx'
    # 'sent_ids', 'tokens', 'piece_idxs', 'token_lens', 'attention_masks',
    # 'entity_label_idxs', 'trigger_label_idxs',
    # 'entity_type_idxs', 'event_type_idxs', 'mention_type_idxs',
    # 'relation_type_idxs', 'role_type_idxs',
    # 'graphs', 'token_nums'
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


class M2E2Dataset(Dataset):
    def __init__(self, image_anno, image_dir, object_pickle, ie_ontology_json, template_json, device='gpu'):
        """
        :param path (str): path to the data file.
        :param gpu (bool): use GPU (default=False).
        """
        self.image_anno = image_anno
        self.image_dir = image_dir
        self.object_pickle = object_pickle
        self.ie_ontology_json = ie_ontology_json
        self.template_json = template_json
        self.data = []
        self.device = device

        self.vocabs, self.templates = self.load_dict_evt()
        self.load_data()

        self.candidate_event, self.candidate_role = self.load_template(self.ie_ontology_json, self.template_json)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_template(ie_ontology_json):
        # since the event type candidates are all the same for the entire dataset, we use a single matrix to represent it
        # candidates
        candidates = inst['candidates']
        candidates_vec = clip.tokenize(candidates).to(self.device)


    def load_dict_evt(self):
        vocabs = dict()
        dict_data = json.load(open(self.ie_ontology_json))
        index = 0
        vocabs = dict()
        # vocab_i2s = dict()
        vocab_s2i = dict()
        templates = list()
        for evt_type in dict_data:
            desp = dict_data[evt_type]

            # vocab_i2s[index] = evt_type
            vocab_s2i[evt_type] = index
            templates.append(desp)

            index += 1

        vocabs['event_type'] = vocab_s2i
        return vocabs, templates


    def load_data(self):
        """Load data from file."""
        # add events annotated
        image_list = list()
        gt_evt_list = list()
        image_id_list = list()

        # add events annotated
        data = json.load(open(self.image_anno))
        for image_id in data:
            inst = data[image_id]
            inst['image_id'] = image_id

            if self.template_choice == 'string':
                inst['candidates'] = self.templates

            self.data.append(inst)

        print('Loaded {} instances from {}'.format(len(self), self.image_anno))
        print('Loaded {} templates from {}'.format(len(self.templates), self.template_choice))


    def numberize(self, vocabs, preprocess):
        event_type_stoi = vocabs['event_type']

        data = []
        for inst in self.data:
            # image
            image_id = inst['image_id']
            image_path = os.path.join(self.image_dir, image_id+'.jpg')
            try:
                image_vec = preprocess(Image.open(image_path)).to(self.device)
            except:
                # # if the image can not be found in the disk, download it
                # image_url = inst['url']
                # img_data = requests.get(image_url).content
                # with open(image_path, 'wb') as handler:
                #     handler.write(img_data)
                # # image_vec = preprocess(Image.open(image_path)).to(self.device)
                pass

            

            # gt event type
            event_type = inst['event_type']  # one image only has one event, currently
            event_type_idx = event_type_stoi[event_type]

            # gt args


            instance = Instance(
                image_id=image_id,
                image_vec=image_vec,
                candidates_vec=candidates_vec,
                event_type_idx=event_type_idx,
                # role_type_idxs=role_type_idxs,
            )
            data.append(instance)
        self.data = data


    def collate_fn(self, batch):
        # batch_event_types = []

        image_ids = [inst.image_id for inst in batch]
        image_vecs = [inst.image_vec for inst in batch]
        candidates_vec = [inst.candidates_vec for inst in batch]

        # for inst in batch:
            # token_num = len(inst.tokens)

            # for classification
            # batch_entity_types.extend(inst.entity_type_idxs +
            #                           [-100] * (max_entity_num - inst.entity_num))
            # batch_event_types.extend(inst.event_type_idxs +
            #                          [-100] * (max_trigger_num - inst.trigger_num))
        
        # only one event type for each image
        batch_event_type = [inst.event_type_idx for inst in batch]
        batch_event_type = torch.LongTensor(batch_event_type).to(self.device)

        batch_image_vecs = torch.LongTensor(image_vecs).to(self.device)
        batch_candidates_vec = torch.LongTensor(candidates_vec).to(self.device)

        return Batch(
            image_id=image_ids,
            image_vec=batch_image_vecs,
            candidates_vec=batch_candidates_vec,
            event_type_idx=batch_event_type,

        )

if __name__ == '__main__':
    dataset = M2E2Dataset(
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
