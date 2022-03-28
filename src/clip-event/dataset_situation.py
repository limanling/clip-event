import codecs
import os

import torch
import torch.utils.data as data
from PIL import Image
from collections import defaultdict
import json
import csv
import pickle
import traceback
import numpy as np
from operator import itemgetter

from src.util import consts
from src.dataflow.numpy.anno_mapping import event_type_norm, role_name_norm


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else consts.UNK_IDX for t in tokens]
    return ids


def map_to_id(token, vocab):
    if token in vocab:
        return vocab[token]
    else:
        return consts.UNK_IDX

def get_labels(class_map_file):
    print('class_map_file', class_map_file)
    label_name = {}
    with open(class_map_file, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            # isArgType = int(row[2])
            # if isArgType:
            if row[2] == '1':
                label_name[row[0]] = row[1]
    print('object_label', label_name)
    return label_name

def load_img_object(img_id, image_dir, transform,
                    load_object=False, object_results=None,
                    object_label=None, object_detection_threshold=.1,
                    vocab_objlabel=None,
                    object_topk=50,):
    try:
        img_path = os.path.join(image_dir, img_id)
        image = Image.open(img_path).convert('RGB')
    except:
        img_path = os.path.join(image_dir, img_id+'.png')
        image = Image.open(img_path).convert('RGB')
    if transform is not None:
        image_vec = transform(image)

    if load_object:
        # load object detection result
        bbox_entities_id = []
        bbox_entities_region = []
        bbox_entities_label = []

        # pad the entire image to bbox vectors, in case the bbox is none
        bbox_entities_id.append('0_0_0_0')
        if transform is not None:
            bbox_entities_region.append(image_vec)
        else:
            bbox_entities_region.append(image)
        bbox_entities_label.append(consts.UNK_IDX)

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
                bbox_entities_id.append(patch_id)
                # bbox_entities_label.append(self.map_to_id(label, self.vocab_situation_noun.word2id))
                bbox_entities_label.append(map_to_id(label, vocab_objlabel))
                if transform is not None:
                    patch_vec = transform(patch)
                    # bbox_id, bbox_vec, label
                    # bbox_entities.append( (patch_id, patch_vec, label) )
                    bbox_entities_region.append(patch_vec)
                else:
                    # bbox_entities.append( (patch_id, patch, label) )
                    bbox_entities_region.append(patch)
                count += 1
            except:
                print('Wrong image ', img_path)
                traceback.print_exc()
        object_num = len(bbox_entities_region)

    if load_object:
        return image, image_vec, bbox_entities_id, bbox_entities_region, bbox_entities_label, object_num
    else:
        return image, image_vec, None, None, None, None


class ImSituDataset(data.Dataset):
    def __init__(self, image_dir, vocab_situation_noun, vocab_situation_role, vocab_situation_verb,
                 event2id, eerole2id,
                 imsitu_ontology_file, imsitu_annotation_file, verb_mapping_file, object_ontology_file=None,
                 object_detection_pkl_file=None, object_detection_threshold=0.2, transform=None, filter_irrelevant_verbs=False,
                 load_object=False, filter_place=False):
        self.image_dir = image_dir
        self.vocab_situation_noun = vocab_situation_noun
        self.vocab_situation_role = vocab_situation_role
        self.vocab_situation_verb = vocab_situation_verb
        self.event2id = event2id
        print('self.event2id', self.event2id)
        self.eerole2id = eerole2id
        print('self.eerole2id', self.eerole2id)


        self.filter_irrelevant_verbs = filter_irrelevant_verbs
        print('************************filter_irrelevant_verbs: ', self.filter_irrelevant_verbs)
        self.filter_place = filter_place
        print('************************filter_place: ', self.filter_place)

        self.load_object = load_object
        if self.load_object:
            self.object_detection_threshold = object_detection_threshold
            self.object_label = get_labels(object_ontology_file)
            self.object_results = pickle.load(open(object_detection_pkl_file, 'rb'))
        else:
            self.object_detection_threshold = 0.2
            self.object_label = None
            self.object_results = None

            # imsitu_info = json.load(open(os.path.join(imsitu_dir, "imsitu_space.json")))
        imsitu_info = json.load(open(imsitu_ontology_file))
        self.nouns = imsitu_info["nouns"]
        self.verbs = imsitu_info["verbs"]
        self.verb_roles = self._load_verb_roles(imsitu_info["verbs"], vocab_situation_verb.word2id, vocab_situation_role.word2id)
        self.role_masks = self._verb_role_mask(self.verb_roles)
        self.annotation = json.load(open(imsitu_annotation_file))

        self.sr_mapping_verb, self.sr_mapping_role = self.load_mapping_all(verb_mapping_file)
        # print('self.sr_mapping_verb', self.sr_mapping_verb)
        # print('self.sr_mapping_role', self.sr_mapping_role)

        # self.img_verbs, self.img_verb_roles, self.img_verb_role_num = self.__getobjects__(image_dir)
        self.img_verbs, self.img_verb_roles = self.__getobjects__(image_dir)
        self.idx_imgs = self.__getids__()
        self.ids = list(self.idx_imgs.keys())
        self.transform = transform
        

    # def load_mapping_verb(self, verb_mapping_file):
    #     verb_type_dict = dict()
    #     for line in open(verb_mapping_file):
    #         line = line.rstrip('\n')
    #         tabs = line.split('\t')
    #         verb_type_dict[tabs[0]] = tabs[1]
    #     return verb_type_dict

    def load_mapping_all(self, verb_mapping_file):
        sr_verb_mapping = defaultdict()
        sr_role_mapping = defaultdict(lambda : defaultdict(str))
        for line in codecs.open(verb_mapping_file, 'r', encoding='utf-8'):
            line = line.rstrip('\n')
            tabs = line.split('\t')
            sr_verb = tabs[0]
            sr_role = tabs[1]
            ee_event = tabs[2]
            ee_role = tabs[3]
            sr_role_mapping[sr_verb][sr_role] = ee_role
            sr_verb_mapping[sr_verb] = ee_event
        return sr_verb_mapping, sr_role_mapping

    def get_sr_mapping(self):
        return self.sr_mapping_verb, self.sr_mapping_role


    def _load_verb_roles(self, verb_info, verb2id, role2id):
        verb_roles = defaultdict(set)
        for verb in verb_info:
            for role in verb_info[verb]['roles']:
                if self.filter_place and 'place' == role.lower():
                    continue
                verb_roles[verb2id[verb]].add(role2id[role])
        return verb_roles

    def _verb_role_mask(self, verb_roles):
        # role_masks = dict()
        # for verb in verb_roles:
        #     indexes = []
        #     for role in verb_roles[verb]:
        #         indexes.append(role2id[role])
        #     i = torch.LongTensor([[0]*len(indexes), indexes])
        #     v = torch.LongTensor([1] * len(indexes))
        #     role_masks[verb2id[verb]] = torch.sparse.FloatTensor(i, v, torch.Size([self.vocab_situation_role.size]))
        # return role_masks
        row_indexes = []
        column_indexes = []
        for verb in verb_roles:
            for role in verb_roles[verb]:
                column_indexes.append(role)
            row_indexes.extend([verb] * len(verb_roles[verb]))
        i = torch.LongTensor([row_indexes, column_indexes])
        v = torch.LongTensor([1] * len(row_indexes))
        role_masks = torch.sparse.FloatTensor(i, v, torch.Size([self.vocab_situation_verb.size, self.vocab_situation_role.size])).requires_grad_(False)
        return role_masks


    def get_role_mask(self):
        return self.role_masks


    def get_verb_role_mapping(self):
        return self.verb_roles




    # img -> patch & entity
    def __getobjects__(self, image_dir):
        img_verbs = defaultdict(lambda: str)
        img_verb_roles = defaultdict(lambda : defaultdict(set))
        # img_verb_role_num = defaultdict(int)
        for image_id in os.listdir(image_dir):
            if image_id not in self.annotation:
                continue
            # if self.load_object and len(self.object_results[image_id]) == 0: #(image_id not in self.object_results or ):
            #     # image does not have any objects
            #     continue
            verb = self.annotation[image_id]['verb']
            if self.filter_irrelevant_verbs and verb not in self.sr_mapping_verb:
                continue

            img_verbs[image_id] = verb.lower()
            # ontology
            # verb_roles[image_id] = self.verbs[verb]['order']
            # role values
            frames = self.annotation[image_id]['frames']
            for frame in frames:
                for role in frame:
                    role = role.lower()
                    if self.filter_place and 'place' == role.lower():
                        continue
                    role_value_id = frame[role]
                    if len(role_value_id) > 0:
                        role_value = self.nouns[role_value_id]['gloss']
                        # lower()
                        img_verb_roles[image_id][role].update(role_value)
                        # img_verb_role_num[image_id] = img_verb_role_num[image_id] + 1

        return img_verbs, img_verb_roles #, img_verb_role_num

    # idx -> img
    def __getids__(self):
        idx_img = {}
        index = 0
        for img in self.img_verbs:
            idx_img[index] = img
            index += 1

        print("number of images: ", len(idx_img))
        return idx_img

    def __getitem__(self, index):
        """Returns one data pair (image, captions, regions and entities)."""
        img_id = self.idx_imgs[index]
        return self.get_img_info(img_id)


    def get_img_info(self, img_id):
        # print(img_id, self.img_verbs[img_id])
        # try:
        #     print(self.sr_mapping_verb[self.img_verbs[img_id]])
        # except:
        #     print('no ace type')
        verb = map_to_id(self.img_verbs[img_id], self.vocab_situation_verb.word2id)
        if self.img_verbs[img_id] in self.sr_mapping_verb:
            event_str = 'B-' + event_type_norm(self.sr_mapping_verb[self.img_verbs[img_id]])
            event = self.event2id[event_str]
        else:
            event = self.event2id[consts.O_LABEL_NAME]

        verb_roles = self.img_verb_roles[img_id]  # role -> role_values
        roles = []
        roles_ee = []
        args = []
        for role in verb_roles:
            # keep the one has the mapping
            if (self.img_verbs[img_id] in self.sr_mapping_role) and (role in self.sr_mapping_role[self.img_verbs[img_id]]):
                role_ee_str = role_name_norm(self.sr_mapping_role[self.img_verbs[img_id]][role])
            else:
                role_ee_str = consts.ROLE_O_LABEL_NAME
            for role_arg in verb_roles[role]:
                # roles.append(self.make_one_hot(role, self.vocab_situation_role.word2id))
                roles.append(map_to_id(role, self.vocab_situation_role.word2id))
                roles_ee.append(self.eerole2id[role_ee_str])  # same vocab???
                args.append(map_to_id(role_arg, self.vocab_situation_noun.word2id))
        arg_num = len(args)

        # roles = np.asarray(roles)

        image, image_vec, bbox_entities_id, bbox_entities_region, bbox_entities_label, object_num \
            = load_img_object(img_id, self.image_dir, self.transform,
                              self.load_object, self.object_results, self.object_label,
                              self.object_detection_threshold,
                              self.vocab_situation_noun.word2id)

        if self.transform is not None:
            if self.load_object:
                return img_id, image_vec, verb, event, roles, roles_ee, args, bbox_entities_id, bbox_entities_region, bbox_entities_label, arg_num, object_num
            else:
                return img_id, image_vec, verb, event, roles, roles_ee, args, None, None, None, arg_num, None
        else:
            if self.load_object:
                return img_id, image, verb, event, roles, roles_ee, args, bbox_entities_id, bbox_entities_region, bbox_entities_label, arg_num, object_num
            else:
                return img_id, image, verb, event, roles, roles_ee, args, None, None, None, arg_num, None

    def __len__(self):
        return len(self.ids)

    # def make_one_hots(self, roles, vocab):
    #     role_vec = np.zeros(len(roles), len(vocab))
    #     for i, r in enumerate(roles):
    #         print(i, r, vocab[r])
    #         role_vec[i][vocab[r]] = 1
    #         # ids = [vocab[t] if t in vocab else consts.UNK_IDX ]
    #     return role_vec

    def make_one_hot(self, role, vocab):
        role_vec = [0] * len(vocab) #np.zeros(len(vocab))
        # print(role)
        # if role not in vocab:
        # print('vocab', vocab)
        role_vec[vocab[role]] = 1
        return role_vec


def image_collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (img_id, captionvec, regions_tensor, entity_tensor). <-- (__getitem__ !!!)

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption/regions/entities (including padding) is not supported in default.
    """
    # Sort a data list by arg_num (descending order).
    batch.sort(key=lambda x: x[-2], reverse=True)
    img_id_batch, image_batch, verb_batch, event_batch, roles_batch, ee_roles_batch, args_batch, bbox_entities_id, bbox_entities_region, bbox_entities_label, arg_num_batch, object_num_batch = zip(*batch)  # zip(['a', 'b', 'c'], [1, 2, 3]) =

    # Merge images (from tuple of 3D tensor to 4D tensor).
    image_batch = torch.stack(image_batch, 0)

    # object mask

    return img_id_batch, image_batch, verb_batch, event_batch, roles_batch, ee_roles_batch, args_batch, bbox_entities_id, bbox_entities_region, bbox_entities_label, arg_num_batch, object_num_batch

def unpack(batch, device, load_object=False):
    img_id_batch = batch[0]
    # print('img_id_batch', img_id_batch)
    # print('device', device)
    image_batch = batch[1].to(device)
    verb_gt_batch = torch.LongTensor(batch[2]).to(device)
    event_gt_batch = torch.LongTensor(batch[3]).to(device)

    arg_num_batch = np.array(batch[-2])
    # roles_gt_batch = torch.LongTensor(batch[3]).to(device)
    # args_batch = torch.LongTensor(batch[4]).to(device)
    arg_num_max = max(arg_num_batch)
    # roles_gt_batch = torch.zeros([len(img_id_batch), arg_num_max, len(batch[3][0][0])]).type(torch.LongTensor).to(device)
    roles_gt_batch = torch.zeros([len(img_id_batch), arg_num_max]).type(torch.LongTensor).to(device)
    ee_roles_gt_batch = torch.zeros([len(img_id_batch), arg_num_max]).type(torch.LongTensor).to(device)
    args_gt_batch = torch.zeros([len(img_id_batch), arg_num_max]).type(torch.LongTensor).to(device)
    for batch_idx, _ in enumerate(batch[4]):
        roles_gt_batch[batch_idx][:arg_num_batch[batch_idx]] = torch.LongTensor(batch[4][batch_idx])
        ee_roles_gt_batch[batch_idx][:arg_num_batch[batch_idx]] = torch.LongTensor(batch[5][batch_idx])
        args_gt_batch[batch_idx][:arg_num_batch[batch_idx]] = torch.LongTensor(batch[6][batch_idx])

    if load_object:
        bbox_entities_id = batch[7]
        object_num_batch = np.array(batch[-1])
        object_num_max = max(object_num_batch)
        bbox_entities_region = torch.zeros(len(img_id_batch), object_num_max, 3, 224, 224).to(device)
        bbox_entities_label = torch.zeros(len(img_id_batch), object_num_max).to(device)
        for b_idx, _ in enumerate(batch[8]):
            for obj_idx, _ in enumerate(batch[8][b_idx]):
                bbox_entities_region[b_idx][obj_idx] = batch[8][b_idx][obj_idx]
                bbox_entities_label[b_idx][obj_idx] = batch[9][b_idx][obj_idx]
        return img_id_batch, image_batch, verb_gt_batch, event_gt_batch, roles_gt_batch, ee_roles_gt_batch, args_gt_batch, \
               bbox_entities_id, bbox_entities_region, bbox_entities_label, \
               arg_num_batch, object_num_batch
    else:
        return img_id_batch, image_batch, verb_gt_batch, event_gt_batch, roles_gt_batch, ee_roles_gt_batch, args_gt_batch, \
               None, None, None, arg_num_batch, None
{"mode":"full","isActive":false}