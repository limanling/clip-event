import enum
import os
import json
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple, defaultdict
import requests
import csv
import pickle
from operator import itemgetter
import traceback

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

from torchvision.datasets.folder import DatasetFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from clip import tokenize

# instance_fields = [
#     'image_id',
#     'image_vec',
#     'description',
#     'description_vec',
#     'labels_per_image',
#     'labels_per_text',
#     'index_description_pos'
# ]
# Instance = namedtuple('Instance', field_names=instance_fields,
#                       defaults=[None] * len(instance_fields))

batch_fields = [
    'image_id',
    'image_vec',
    'description',
    'description_vec',
    'labels_per_image',
    'labels_per_text',
    'index_description_pos',
    'object_id', 
    'object_vec', 
    'object_label', 
    'object_num',
    'entitytxt_id', 
    'entitytxt_vec', 
    'entitytxt_label', 
    'entitytxt_num',
    'eventtxt_id', 
    'eventtxt_vec', 
    'eventtxt_label', 
    'eventtxt_num',
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


class VOADataset(Dataset):
    def __init__(self, image_caption_json_list, image_dir_list, preprocess=None, tokenize=None, device='gpu'):
        """
        :param path (str): path to the data file.
        :param gpu (bool): use GPU (default=False).
        """
        self.image_caption_json_list = image_caption_json_list
        self.image_dir_list = image_dir_list
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

        # load image-caption pairs
        for image_caption_json, image_dir in zip(self.image_caption_json_list, self.image_dir_list):
            data = json.load(open(image_caption_json))
            for doc_id in data:
                for image_idx in data[doc_id]:
                    image_id = '%s_%s' % (doc_id, image_idx)
                    inst = dict()
                    inst['image_id'] = image_id
                    inst['image_dir'] = image_dir
                    inst['url'] = data[doc_id][image_idx]['url']
                    inst['caption'] = data[doc_id][image_idx]['cap'].replace('FILE - ', '')
                    # context can not exceed 77 length
                    # ==> clip.py, tokenize()

                    self.data.append(inst)
        

        logging.info('Loaded {} instances from {}'.format(len(self), self.image_caption_json_list))

    def clean_imageid(self, image_id):
        return image_id.replace('.', '_')

    def collate_fn(self, batch): #, preprocess, tokenize):
        # print('batch', batch[0]['image_id'])    

        # image_ids_batch = [self.clean_imageid(inst['image_id']) for inst in batch]
        image_ids_batch = list()
        image_vecs_batch = list()
        for inst in batch:
            image_id = self.clean_imageid(inst['image_id'])
            image_ids_batch.append(image_id)

            image_dir = inst['image_dir']
            image_path = os.path.join(image_dir, image_id+'.jpg')
            try:
                image_vec = self.preprocess(Image.open(image_path))
                image_vec = image_vec.to(self.device)
            except:
                # if the image can not be found in the disk, download it
                image_url = inst['url']
                img_data = requests.get(image_url).content
                with open(image_path, 'wb') as handler:
                    handler.write(img_data)
                image_vec = self.preprocess(Image.open(image_path))
                image_vec = image_vec.to(self.device)
            image_vecs_batch.append(image_vec)
        image_vecs_batch = torch.stack(image_vecs_batch, dim=0).to(self.device)
        
        descriptions_batch = [inst['caption'] for inst in batch]
        description_vecs_batch = self.tokenize(descriptions_batch).to(self.device)

        labels_per_image_batch = torch.arange(len(image_ids_batch)).to(self.device)
        labels_per_text_batch = torch.arange(len(image_ids_batch)).to(self.device)

        return Batch(
            image_id=image_ids_batch,
            image_vec=image_vecs_batch,
            description=descriptions_batch,
            description_vec=description_vecs_batch,
            labels_per_image=labels_per_image_batch,
            labels_per_text=labels_per_text_batch,
            index_description_pos=torch.arange(len(image_ids_batch)).to(self.device)
        )


def map_to_id(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        return consts.UNK_IDX

def get_object_labels(class_map_file):
    # print('class_map_file', class_map_file)
    label_name = {}
    with open(class_map_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # isArgType = int(row[2])
            # if isArgType:
            if row[2] == '1':
                label_name[row[0]] = row[1]
    logging.info('Allowed object labels: %s' % str(label_name))
    return label_name

def load_img_object(img_id, img_path, transform=None,
                    load_object=False, object_results=None,
                    object_label=None, object_detection_threshold=.1,
                    vocab_objlabel=None,
                    object_topk=50,):
    image = Image.open(img_path).convert('RGB')
    
    if transform is not None:
        image_vec = transform(image)

    if load_object:
        # load object detection result
        objects_id = []
        objects_region = []
        objects_label = []

        # pad the entire image to bbox vectors, in case the bbox is none
        objects_id.append('0_0_0_0')
        if transform is not None:
            objects_region.append(image_vec)
        else:
            objects_region.append(image)
        if vocab_objlabel is None:
            objects_label.append('UNKNOWN')
        else:
            objects_label.append(consts.UNK_IDX)

        if img_id in object_results:
            objects = object_results[img_id]
            objects_by_score = sorted(objects, key=itemgetter('score'))
            count = 1
            for object in objects_by_score:
                if count > object_topk:
                    break
                if object['label'] not in object_label:
                    # print('rejected labels', object['label'])
                    continue
                # print('accepted labels', object_label[object['label']])
                label = object_label[object['label']]
                bbox = object['bbox']
                score = object['score']
                if score < object_detection_threshold:
                    continue
                # transform patch to patch_vec
                try:
                    patch = image.crop(bbox)
                    patch_id = '%d_%d_%d_%d' % (bbox[0], bbox[1], bbox[2], bbox[3])
                    objects_id.append(patch_id)
                    if vocab_objlabel is None:
                        objects_label.append(label)
                    else:
                        objects_label.append(map_to_id(label, vocab_objlabel))
                    if transform is not None:
                        patch_vec = transform(patch)
                        objects_region.append(patch_vec)
                    else:
                        objects_region.append(patch)
                    count += 1
                except:
                    print('Wrong image ', img_path)
                    traceback.print_exc()
        
        object_num = len(objects_region)

    if load_object:
        return image, image_vec, objects_id, objects_region, objects_label, object_num
    else:
        return image, image_vec, None, None, None, None

def parse_offset_str(offset_str):
    docid = offset_str[:offset_str.rfind(':')]
    start = int(offset_str[offset_str.rfind(':') + 1:offset_str.rfind('-')])
    end = int(offset_str[offset_str.rfind('-') + 1:])
    return docid, start, end

# def get_trigger_context(ltf_dir, docid, start, end):
#     tokens = []
#     tokens_trigger = []

#     ltf_file_path = os.path.join(ltf_dir, docid + '.ltf.xml')
#     if not os.path.exists(ltf_file_path):
#         return '[ERROR]NoLTF %s' % docid
#     tree = ET.parse(ltf_file_path)
#     root = tree.getroot()
#     for doc in root:
#         for text in doc:
#             for seg in text:
#                 seg_beg = int(seg.attrib["start_char"])
#                 seg_end = int(seg.attrib["end_char"])
#                 if start >= seg_beg and end <= seg_end:
#                     for token in seg:
#                         if token.tag == "TOKEN":
#                             tokens.append(token.text)
#                             token_beg = int(token.attrib["start_char"])
#                             token_end = int(token.attrib["end_char"])
#                             if start <= token_beg and end >= token_end:
#                                 tokens_trigger.append(token.text)
#                 if len(tokens) > 0:
#                     return ' '.join(tokens_trigger), ' '.join(tokens)
#     return ' '.join(tokens_trigger), ' '.join(tokens)

def load_entity_cs(entity_cs, entities, entity_type, entity_name, entity_mentions, load_mention=False):
    for line in open(entity_cs):
        line = line.rstrip('\n')
        tabs = line.split('\t')
        if line.startswith(':Entity'):
            if 'type' == tabs[1]:
                entity_type[tabs[0]] = tabs[2].split('#')[-1]
                # if len(tabs) == 4:
                #     entity_types[tabs[0]][tabs[2].split('#')[-1]] = float(tabs[3])
                # else:
                #     entity_types[tabs[0]][tabs[2].split('#')[-1]] = 1.0
            elif 'canonical_mention' in tabs[1]:
                offset = tabs[3]
                docid = offset.split(':')[0]
                entity_name[tabs[0]] = tabs[2][1:-1]
                conf = tabs[4]
                entities[docid][tabs[0]] = conf
                # docs.add(docid)
            elif 'mention' in tabs[1]:
                if load_mention:
                    offset = tabs[3]
                    entity_mentions[tabs[0]].add(offset)
    return entities, entity_type, entity_name, entity_mentions

def load_event_cs(event_cs, events, event_type, event_mentions, event_arguments):
    for line in open(event_cs):
        if line.startswith(':Event'):
            line = line.rstrip('\n')
            tabs = line.split('\t')
            event_id = tabs[0]
            if tabs[1] == 'type':
                event_type[event_id] = tabs[2].split('#')[-1]
            elif 'mention' in tabs[1]:
                offset = tabs[3]
                docid = offset.split(':')[0]
                event_mentions[event_id].add(tabs[2][1:-1])
                events[docid][event_id] = event_type[event_id]
                # docs.add(docid)
            elif len(tabs[1]) == 2:
                event_4tuple[event_id][tabs[1]] = tabs[2]
            elif tabs[1].endswith('_Time.actual'):
                event_4tuple[event_id]['time'] = tabs[2]
            elif 'mention' not in tabs[1] and '_' in tabs[1]:
                # arg roles
                docid = tabs[3].split(':')[0]
                typestr = tabs[1]
                arg_role = typestr[typestr.rfind('_')+1:].split('.')[0]
                arg_id = tabs[2]
                # arg_offset = tabs[3]
                event_arguments[event_id][arg_role].add(arg_id)
                # event_mention_arguments[offset][arg_role].add(arg_id)



def load_ie_cs(input_entities, input_fillers=None, input_events=None, input_temporal_orders=None, load_mention=False):
    # docid -> entityid -> confidence
    doc_entities = defaultdict(lambda : defaultdict(float))
    # entityid -> canonical_mention
    entity_name = defaultdict(str)
    # entityid -> all mentions
    entity_mentions = defaultdict(set)
    # entity_types: entityid -> type -> confidence
    # entity_types = defaultdict(lambda : defaultdict(float)) 
    # entityid -> type
    entity_type = dict()
    # eventid -> type
    event_type = dict()
    # docid -> eventid -> confidence
    doc_events = defaultdict(lambda : defaultdict(str))
    # eventid -> canonical_mention
    event_mentions = defaultdict(set)
    # eventid -> arg_role -> arg_id
    event_arguments = defaultdict(lambda : defaultdict(set))

    # TODO: load relations

    try:
        if input_entities:
            for input_entity in input_entities:
                load_entity_cs(input_entity, doc_entities, entity_type, entity_name, entity_mentions, load_mention=load_mention)
        if input_events:
            for input_event in input_events:
                load_event_cs(input_event, doc_events, event_type, event_mentions, event_arguments)
    except Exception as e:
        traceback.print_exc() 

    return doc_entities, entity_type, entity_name, entity_mentions, doc_events, event_type, event_mentions, event_arguments


class VOADescriptionDataset(Dataset):
    def __init__(self, 
                # description
                posneg_descriptions_json, 
                # image
                image_caption_json_list, image_dir_list, 
                # text ie
                load_ie=False, ie_ontology_json=None, ltf_dir=None, input_entities=None, input_fillers=None, input_relations=None, input_events=None, input_temporal_orders=None, 
                # object
                load_object=False, object_pickle=None, object_ontology_file=None, object_detection_threshold=0.2, object_topk=50, 
                # situation recognition
                load_sr=False, 
                constrastive_overbatch=True, constrative_loss='ce', 
                preprocess=None, tokenize=None, device=torch.device('cuda')):
        self.posneg_descriptions_json = posneg_descriptions_json
        self.image_caption_json_list = image_caption_json_list
        self.image_dir_list = image_dir_list
        self.ie_ontology_json = ie_ontology_json
        self.data = []

        self.device = device
        self.preprocess = preprocess
        self.tokenize = tokenize

        # self.vocabs, self.templates = self.load_dict_evt()
        self.load_data()
        self.description_num_pos = len(self.data[0]['pos'])
        self.description_num_neg = len(self.data[0]['neg_event']) + len(self.data[0]['neg_argument'])
        self.description_num = len(self.data[0]['pos']) + len(self.data[0]['neg_event']) + len(self.data[0]['neg_argument'])
        logging.info('Each instance has {} instances'.format(self.description_num))
        self.constrastive_overbatch = constrastive_overbatch
        self.constrative_loss = constrative_loss

        self.load_object = load_object
        self.object_detection_threshold = object_detection_threshold
        self.object_topk = object_topk
        if self.load_object:
            self.object_label = get_object_labels(object_ontology_file)
            self.object_results = dict()
            for _ in object_pickle:
                self.object_results.update(pickle.load(open(_, 'rb')))
        else:
            self.object_label = None
            self.object_results = None

        self.load_ie = load_ie
        if self.load_ie:
            self.doc_entities, self.entity_type, self.entity_name, self.entity_mentions, self.doc_events, self.event_type, self.event_mentions, self.event_arguments = load_ie_cs(input_entities=input_entities, input_fillers=input_fillers, input_events=input_events, input_temporal_orders=input_temporal_orders, load_mention=False)
            # print(len(self.doc_entities), len(self.doc_events))


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

        # load description 
        posneg_descriptions = json.load(open(self.posneg_descriptions_json))

        # load image-caption pairs
        for image_caption_json, image_dir in zip(self.image_caption_json_list, self.image_dir_list):
            data = json.load(open(image_caption_json))
            for doc_id in data:
                for image_idx in data[doc_id]:
                    image_id = '%s_%s' % (doc_id, image_idx)
                    image_id = self.clean_imageid(image_id)
                    inst = dict()
                    inst['image_id'] = image_id
                    if image_id not in posneg_descriptions:
                        continue
                    inst['image_dir'] = image_dir
                    inst['url'] = data[doc_id][image_idx]['url']
                    inst['caption'] = data[doc_id][image_idx]['cap'].replace('FILE - ', '')
                    # context can not exceed 77 length
                    # ==> clip.py, tokenize()

                    # generate the postive description
                    # FIXIT: use caption as positive
                    # inst['pos'] = [inst['caption']] 
                    inst['pos'] = posneg_descriptions[image_id]['pos']

                    # generate the negative description
                    inst['neg_event'] = posneg_descriptions[image_id]['neg_event']
                    if 'neg_argument' in posneg_descriptions[image_id]:
                        inst['neg_argument'] = posneg_descriptions[image_id]['neg_argument']
                    else:
                        inst['neg_argument'] = posneg_descriptions[image_id]['neg_event']
                    
                    self.data.append(inst)

        logging.info('Loaded {} instances from {}'.format(len(self), self.image_caption_json_list))

    def clean_imageid(self, image_id):
        return image_id.replace('.', '_')

    def collate_fn(self, batch): #, preprocess, tokenize):  

        BATCH_SIZE = len(batch)
        
        # ==================== Create image (and object) vectors ==================== 
        # image_ids_batch = [self.clean_imageid(inst['image_id']) for inst in batch]
        image_ids_batch = list()
        image_vecs_batch = list()
        object_ids_batch = None
        object_vecs_batch = None
        object_labels_batch = None
        object_num_batch = None
        if self.load_object:
            object_ids_batch = list()
            object_vecs_list = list()
            object_labels_list = list()
            object_num_list = list()
        for inst in batch:
            # image_id = self.clean_imageid(inst['image_id'])
            image_id = inst['image_id']
            image_ids_batch.append(image_id)

            image_dir = inst['image_dir']
            image_path = os.path.join(image_dir, image_id+'.jpg')

            try:
                # image_vec = self.preprocess(Image.open(image_path)).to(self.device)
                image, image_vec, objects_id, objects_region, objects_label, object_num \
                    = load_img_object(image_id, image_path, transform=self.preprocess,
                    load_object=self.load_object, object_results=self.object_results,
                    object_label=self.object_label, object_detection_threshold=self.object_detection_threshold,
                    vocab_objlabel=None,
                    object_topk=self.object_topk)
            except:
                # if the image can not be found in the disk, download it
                image_url = inst['url']
                img_data = requests.get(image_url).content
                with open(image_path, 'wb') as handler:
                    handler.write(img_data)
                # image_vec = self.preprocess(Image.open(image_path)).to(self.device)
                image, image_vec, objects_id, objects_region, objects_label, object_num \
                    = load_img_object(image_id, image_path, transform=self.preprocess,
                    load_object=self.load_object, object_results=self.object_results,
                    object_label=self.object_label, object_detection_threshold=self.object_detection_threshold,
                    vocab_objlabel=None,
                    object_topk=self.object_topk)
                
            image_vecs_batch.append(image_vec)
            if self.load_object:
                object_ids_batch.append(objects_id)
                object_vecs_list.append(objects_region)
                object_labels_list.append(objects_label)
                object_num_list.append(object_num)
        image_vecs_batch = torch.stack(image_vecs_batch, dim=0).to(self.device)
        if self.load_object:
            # pad to object_num_max
            object_num_max = max(object_num_list)
            object_vecs_batch = torch.zeros(BATCH_SIZE, object_num_max, 3, 224, 224).to(self.device)
            # object_labels_batch = torch.zeros(BATCH_SIZE, object_num_max).to(self.device)
            object_labels_batch = object_labels_list
            # object_num_batch = torch.LongTensor(object_num_list).to(self.device)
            object_num_batch = torch.zeros(BATCH_SIZE, object_num_max).long().to(self.device)
            for batch_idx, _ in enumerate(object_labels_list):
                for obj_idx, _ in enumerate(object_labels_list[batch_idx]):
                    object_vecs_batch[batch_idx][obj_idx] = object_vecs_list[batch_idx][obj_idx]
                    # object_labels_batch[batch_idx][obj_idx] = object_labels_list[batch_idx][obj_idx]
                    object_num_batch[batch_idx][obj_idx] = 1

        # ==================== Create IE vectors ==================== 
        entitytxt_ids_batch = None
        entitytxt_vecs_batch = None
        entitytxt_labels_batch = None
        entitytxt_num_batch = None
        eventtxt_ids_batch = None
        eventtxt_vecs_batch = None
        eventtxt_labels_batch = None
        eventtxt_num_batch = None
        if self.load_ie:
            entitytxt_ids_batch = list()
            entitytxt_vecs_list = list()
            entitytxt_labels_list = list()
            entitytxt_num_list = list()
            for image_id in image_ids_batch:
                entitytxt_ids_batch.append([entity_id for entity_id in self.doc_entities[image_id]])
                entitytxt_names = [self.entity_name[entity_id] for entity_id in self.doc_entities[image_id]]
                entitytxt_vecs_list.append(self.tokenize(entitytxt_names).to(self.device))
                entitytxt_labels_list.append([self.entity_type[entity_id] for entity_id in self.doc_entities[image_id]])
                entitytxt_num_list.append(len(self.doc_entities[image_id]))
            # pad to entitytxt_num_max
            entitytxt_num_max = max(entitytxt_num_list)
            entitytxt_vecs_batch = torch.zeros(BATCH_SIZE, entitytxt_num_max, 77).long().to(self.device)
            # entitytxt_labels_batch = torch.zeros(BATCH_SIZE, entitytxt_num_max).long().to(self.device)
            entitytxt_labels_batch = entitytxt_labels_list
            # entitytxt_num_batch = torch.LongTensor(entitytxt_num_list).to(self.device)
            entitytxt_num_batch = torch.zeros(BATCH_SIZE, entitytxt_num_max).long().to(self.device)
            for batch_idx, _ in enumerate(entitytxt_labels_list):
                for obj_idx, _ in enumerate(entitytxt_labels_list[batch_idx]):
                    entitytxt_vecs_batch[batch_idx][obj_idx] = entitytxt_vecs_list[batch_idx][obj_idx]
                    # entitytxt_labels_batch[batch_idx][obj_idx] = entitytxt_labels_list[batch_idx][obj_idx]
                    entitytxt_num_batch[batch_idx][obj_idx] = 1
                
            
            # TODO: load argument edges
            eventtxt_ids_batch = list()
            eventtxt_vecs_list = list()
            eventtxt_labels_list = list()
            eventtxt_num_list = list()
            for image_id in image_ids_batch:
                eventtxt_ids_batch.append([event_id for event_id in self.doc_events[image_id]])
                eventtxt_names = [', '.join(self.event_mentions[event_id]) for event_id in self.doc_events[image_id]]
                eventtxt_vecs_list.append(self.tokenize(eventtxt_names).to(self.device))
                eventtxt_labels_list.append([self.event_type[event_id] for event_id in self.doc_events[image_id]])
                eventtxt_num_list.append(len(self.doc_events[image_id]))
            # pad to eventtxt_num_max
            eventtxt_num_max = max(eventtxt_num_list)
            eventtxt_vecs_batch = torch.zeros(BATCH_SIZE, eventtxt_num_max, 77).long().to(self.device)
            # eventtxt_labels_batch = torch.zeros(BATCH_SIZE, eventtxt_num_max).long().to(self.device)
            eventtxt_labels_batch = eventtxt_labels_list
            # eventtxt_num_batch = torch.LongTensor(eventtxt_num_list).to(self.device)
            eventtxt_num_batch = torch.zeros(BATCH_SIZE, eventtxt_num_max).long().to(self.device)
            for batch_idx, _ in enumerate(eventtxt_labels_list):
                for obj_idx, _ in enumerate(eventtxt_labels_list[batch_idx]):
                    eventtxt_vecs_batch[batch_idx][obj_idx] = eventtxt_vecs_list[batch_idx][obj_idx]
                    # eventtxt_labels_batch[batch_idx][obj_idx] = eventtxt_labels_list[batch_idx][obj_idx]
                    eventtxt_num_batch[batch_idx][obj_idx] = 1
            
        
        # ==================== Create text vectors ==================== 
        # descriptions_batch = [inst['pos'][0] for inst in batch]
        descriptions_batch = list()
        for inst in batch:
            descriptions_batch.extend(inst['pos'])
            descriptions_batch.extend(inst['neg_event'])
            descriptions_batch.extend(inst['neg_argument'])
        description_vecs_batch = self.tokenize(descriptions_batch).to(self.device)

        # ==================== Create label vectors ==================== 
        if self.constrative_loss == 'ce': 
            if self.description_num_pos == 1:
                if self.constrastive_overbatch:
                    # constrastive loss over batch
                    # the positive text samples are [0,3,6,9] when description_num = 3 (batch_size, )
                    labels_per_image_batch = torch.arange(len(image_ids_batch)).to(self.device)
                    labels_per_image_batch = labels_per_image_batch * self.description_num
                else:
                    # constrastive loss over instance
                    # the first text of each instance is the positive text sample [0,0,0,0] (batch_size, )
                    labels_per_image_batch = torch.zeros(len(image_ids_batch)).long().to(self.device)
            else:
                raise RuntimeError("Invalid constrastive_loss '{}' when loading data. Only constrative_loss=CrossEntropyLoss is allowed when description_num_pos > 1. ".format(self.constrative_loss))
        elif self.constrative_loss == 'bce':
            if self.constrastive_overbatch:
                # constrastive loss over batch
                # the positive text samples are [[1,1,0,0, 0,0,0,0],[0,0,0,0, 1,1,0,0]] when image_num=2, description_num = 4 (description_num_pos=2, description_num_neg=2) (batch_size, batch_size*description_num)
                raise RuntimeError("Set constrastive_overbatch=false for constrative_loss=='bce'.") 
            else:
                # constrastive loss over instance
                # the positive text samples are [[1,1,0,0],[1,1,0,0]] when image_num=2, description_num = 4 (description_num_pos=2, description_num_neg=2) (batch_size, description_num)
                labels_per_image_batch = [[1.] * self.description_num_pos + [0.] * self.description_num_neg for _ in batch]
                labels_per_image_batch = torch.tensor(labels_per_image_batch).to(self.device)
                # labels_per_image_batch = torch.flatten(labels_per_image_batch)
        elif self.constrative_loss == 'kl':
            if self.constrastive_overbatch:
                # the positive text samples are [[1,1,0,0, 0,0,0,0],[0,0,0,0, 1,1,0,0]] when image_num=2, description_num = 4 (description_num_pos=2, description_num_neg=2) (batch_size, batch_size*description_num)
                labels_per_image_batch = torch.zeros()

            else:
                raise RuntimeError("Set constrastive_overbatch=true for constrative_loss=='kl'.") 
        else:
            raise RuntimeError("Invalid constrastive_loss '{}' when loading data. ".format(self.constrative_loss)) 
        
        # labels_per_text is always overbatch, since each instance only has one image, it needs a batch for contrastive learning
        # the positive image samples are [0,0,0,1,1,1,2,2,2,3,3,3] when description_num=3 (batch_size*description_num, )
        # the positive image samples are [0,0,-,-,1,1,-,-,2,2,-,-,3,3,-,-] when description_num = 4 (description_num_pos=2, description_num_neg=2)
        labels_per_text_batch = torch.arange(len(image_ids_batch)).to(self.device)
        labels_per_text_batch = labels_per_text_batch.unsqueeze(1) # 
        labels_per_text_batch = labels_per_text_batch.expand(len(image_ids_batch), self.description_num)
        labels_per_text_batch = labels_per_text_batch.flatten()
        # print('labels_per_text_batch', labels_per_text_batch)

        # mask_description_pos = [1,1,0,0 , 1,1,0,0] when image_num=2, description_num = 4 (description_num_pos=2, description_num_neg=2) (batch_size, description_num)
        mask_description_pos = [[1] * self.description_num_pos + [0] * self.description_num_neg for _ in batch]
        mask_description_pos = torch.LongTensor(mask_description_pos).to(self.device)
        mask_description_pos = torch.flatten(mask_description_pos)
        # index_description_pos = [0,1,4,5]
        index_description_pos = torch.nonzero(mask_description_pos).flatten()
        # print('mask_description_pos', mask_description_pos)
        # print('index_description_pos', index_description_pos)

        return Batch(
            image_id=image_ids_batch,
            image_vec=image_vecs_batch,
            description=descriptions_batch,
            description_vec=description_vecs_batch,
            labels_per_image=labels_per_image_batch,
            labels_per_text=labels_per_text_batch,
            index_description_pos=index_description_pos,
            object_id=object_ids_batch,
            object_vec=object_vecs_batch,
            object_label=object_labels_batch,
            # NOTE: added the image at the first position (to avoid empty vector), so the number is actually object_num+1
            object_num=object_num_batch,
            entitytxt_id=entitytxt_ids_batch,
            entitytxt_vec=entitytxt_vecs_batch,
            entitytxt_label=entitytxt_labels_batch,
            entitytxt_num=entitytxt_num_batch,
            eventtxt_id=eventtxt_ids_batch,
            eventtxt_vec=eventtxt_vecs_batch,
            eventtxt_label=eventtxt_labels_batch,
            eventtxt_num=eventtxt_num_batch
        )

if __name__ == '__main__':
    # working_dir = '/shared/nas/data/m1/manling2/clip-event'
    working_dir = '/home/t-manlingli/clip-event'
    data_dir = 'data/VOA_EN_NW_2017_sample50'
    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    dataset = VOADescriptionDataset(
        # description
        posneg_descriptions_json=os.path.join(working_dir, "data/voa/ie/descriptions_template_template.json"), 
        # image
        # image_caption_json_list='/shared/nas/data/m1/manling2/public_html/voa_data/source_data/image_caption_mapping.json', 
        image_caption_json_list=[os.path.join(working_dir, 'data/voa/small/image_caption_mapping_small.json')], 
        image_dir_list=[os.path.join(working_dir, 'data/VOA_EN_NW_2017_sample50/vision/data/jpg/jpg')], 
        # text ie
        load_ie=True, 
        ie_ontology_json=None, 
        input_entities = [
            os.path.join(working_dir, data_dir, 'edl', 'merged.cs'),     
        ],
        # input_relations = [],
        input_events = [
            os.path.join(working_dir, data_dir, 'event', 'event_rewrite.cs')#'event_coref_timesimple.cs') #'event_coref.cs')] #'event_coref_fix.cs')]
        ], 
        ltf_dir = os.path.join(working_dir, data_dir, 'ltf'),
        # object
        load_object=True, 
        object_pickle=[os.path.join(working_dir, 'data/VOA_EN_NW_2017_sample50/cu_objdet_results/det_results_merged_34a.pkl')],  
        object_ontology_file=os.path.join(working_dir, 'config/class-descriptions-boxable.csv'), 
        object_detection_threshold=0.2, object_topk=50, 
        # situation recognition
        load_sr=False, 
        constrastive_overbatch=True, constrative_loss='ce', 
        preprocess=transform, tokenize=tokenize, device=torch.device('cuda')
    )
    loader = DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=dataset.collate_fn
    )

    for batch_idx, batch in enumerate(loader):
        # print(batch)
        image_id = batch.image_id
        description = batch.description
        object_id=batch.object_id
        object_vec=batch.object_vec
        object_label=batch.object_label
        object_num=batch.object_num
        entitytxt_id=batch.entitytxt_id
        entitytxt_vec=batch.entitytxt_vec
        entitytxt_label=batch.entitytxt_label
        entitytxt_num=batch.entitytxt_num
        eventtxt_id=batch.eventtxt_id
        eventtxt_vec=batch.eventtxt_vec
        eventtxt_label=batch.eventtxt_label
        eventtxt_num=batch.eventtxt_num

        print('image_id', image_id)
        print('object_id', object_id)
        print('object_vec', object_vec.size())
        print('object_label', object_label)
        print('object_num', object_num) # added the image at the first position, so the number is actually object_num+1
        print('entitytxt_id', entitytxt_id)
        print('entitytxt_vec', entitytxt_vec.size())
        print('entitytxt_label', entitytxt_label)
        print('entitytxt_num', entitytxt_num)
        print('eventtxt_id', eventtxt_id)
        print('eventtxt_vec', eventtxt_vec.size())
        print('eventtxt_label', eventtxt_label)
        print('eventtxt_num', eventtxt_num)

        print((eventtxt_num == 0).long())

        break