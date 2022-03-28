import enum
import os
import json
import csv
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple, defaultdict
from operator import itemgetter
import traceback

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 933120000

from torchvision.datasets.folder import DatasetFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from clip import tokenize
from utils_image import normalize_bbox

instance_fields = [
    'image_id',
    'image_path',
    'image_vec',
    'name_verb',
    'desc_verb',
    'desc_verb_vec',
    'role_argbbox',
    'argbbox',
    # 'argbbox_vec',
    'name_argrole',
    'desc_argrole',
    'desc_argrole_vec',
    'name_argtype', # 'name_argtype_canonical'
    'name_argtype_all',
    'obj_id',
    'obj_vec',
    'objlabel',
    'objlabel_vec',
    'obj_num'
]
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))

batch_fields = [
    'image_id',
    'image_path',
    'image_vec',
    'name_verb',
    'desc_verb',
    'desc_verb_vec',
    'role_argbbox',
    'argbbox',
    # 'argbbox_vec',
    'name_argrole',
    'desc_argrole',
    'desc_argrole_vec',
    'name_argtype',
    'name_argtype_all',
    'obj_id',
    'obj_vec',
    'objlabel',
    'objlabel_vec',
    'obj_num'
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


class GSRDataset(Dataset):
    def __init__(self, anno_json, image_dir, ontology_json, object_detection, object_class_map, object_threshold=0.2, object_topk=40, load_object=False, prompt=None, preprocess=None, tokenize=None, device='gpu'):
        """
        :param path (str): path to the data file.
        :param gpu (bool): use GPU (default=False).
        """
        self.anno_json = anno_json
        self.image_dir = image_dir
        self.ontology_json = ontology_json
        self.prompt = prompt
        self.data = []

        self.device = device
        self.preprocess = preprocess
        self.tokenize = tokenize
        
        self.load_object = load_object
        if load_object:
            self.object_threshold = object_threshold
            self.object_topk = object_topk
            self.object_label_map = self.get_labels(object_class_map)
            self.object_results = pickle.load(open(object_detection, 'rb'))

        # self.vocabs, self.templates = self.load_dict_evt()
        # {"tattooing": {"framenet": "Create_physical_artwork", "abstract": "AGENT tattooed TARGET with TOOL in PLACE", "def": "to mark the skin with permanent colors and patterns", "order": ["agent", "target", "tool", "place"], "roles": {"tool": {"framenet": "instrument", "def": "The tool used"}, "place": {"framenet": "place", "def": "The location where the tattoo event is happening"}, "target": {"framenet": "representation", "def": "The entity being tattooed"}, "agent": {"framenet": "creator", "def": "The entity doing the tattoo action"}}}, "raining": {"framenet": "Precipitation", "abstract": "it rains in the PLACE", "def": "rain falls.", "order": ["place"], "roles": {"place": {"framenet": "place", "def": "The location where the rain event is happening"}}}
        self.ontology_verbs = json.load(open(ontology_json))['verbs']  
        # "n03024882": {"gloss": ["choker", "collar", "dog collar", "neckband"], "def": "necklace that fits tightly around a woman's neck"},
        self.ontology_nouns = json.load(open(ontology_json))['nouns']
        self.load_data()

        self.candidate_verbs, self.candidate_verb_vecs, self.verb_id2str, self.verb_str2id = self.load_candidate_verbs()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        return self.data[item]


    def load_data(self):
        """Load data from file."""

        # load qa pairs
        data = json.load(open(self.anno_json))
        for image_id in data:
            verb = data[image_id]['verb']
            verb_desc = self.get_verb_desc(verb)
            height = data[image_id]['height']
            width = data[image_id]['width']
            args = data[image_id]['bb']
            arg_bboxes = list()
            arg_role_name = list()
            arg_role_desc = list()
            arg_type_name = list()
            role_argbbox = dict()
            # find the most common noun as obj type
            arg_type_name_dict = defaultdict(lambda: Counter())
            for frame in data[image_id]['frames']:
                for arg_role in frame:
                    arg_name_frameid = frame[arg_role]
                    if len(arg_name_frameid) > 0:
                        arg_name = self.ontology_nouns[arg_name_frameid]['gloss']
                        arg_desc = self.ontology_nouns[arg_name_frameid]['def']
                        arg_type_name_dict[arg_role].update(arg_name)
            for arg_role in args:
                arg_bbox = args[arg_role]
                # # (1) have bbox (2) have obj type annotation
                # if arg_bbox[0] != -1 and len(arg_type_name_dict[arg_role]) > 0:
                #     if arg_bbox[3] == arg_bbox[1]:
                #         arg_bbox[3] += 2
                #     if arg_bbox[2] == arg_bbox[0]:
                #         arg_bbox[2] += 2
                #     arg_bboxes.append(arg_bbox)
                #     arg_role = arg_role.replace('sources', 'source')
                #     arg_role_desc.append(self.get_rolename_desc(verb, arg_role, arg_type_name_dict[arg_role].most_common()[0][0]))
                #     arg_role_name.append(arg_role)
                #     arg_type_name.append(arg_type_name_dict[arg_role].most_common()[0][0])
                arg_role = arg_role.replace('sources', 'source')
                if len(arg_type_name_dict[arg_role]) > 0:
                    arg_type_name.append(arg_type_name_dict[arg_role].most_common()[0][0])
                    arg_role_desc.append(self.get_rolename_desc(verb, arg_role, arg_type_name_dict[arg_role].most_common()[0][0]))
                else:
                    arg_type_name.append(None)
                    arg_role_desc.append(self.get_rolename_desc(verb, arg_role, None))
                arg_role_name.append(arg_role)
                
                if arg_bbox[0] != -1:
                    # fix the bounding boxes
                    if arg_bbox[3] == arg_bbox[1]:
                        arg_bbox[3] += 2
                    if arg_bbox[2] == arg_bbox[0]:
                        arg_bbox[2] += 2
                    arg_bbox = normalize_bbox(arg_bbox, width=width, height=height)
                    arg_bboxes.append(arg_bbox)
                else:
                    arg_bboxes.append(None)
                
                role_argbbox[arg_role] = arg_bbox

            if self.load_object:
                # load object detection result
                objects = self.object_results[image_id]
                objects_by_score = sorted(objects, key=itemgetter('score'))
                count = 1
                objid_list = []
                objbbox_list = []
                objlabel_list = []

                for object in objects_by_score:
                    if count > self.object_topk:
                        break
                    if object['label'] not in self.object_label_map:
                        # print('rejected labels', object['label'])
                        continue
                    # print('accepted labels', object_label[object['label']])
                    objlabel = self.object_label_map[object['label']]
                    objbbox = list(object['bbox_normalized']) #object['bbox']
                    objscore = object['score']
                    if objscore < self.object_threshold:
                        continue
                    # objbboxnorm = normalize_bbox(img_height, img_width, bbox)
                    objid = '%d_%d_%d_%d' % (object['bbox'][0], object['bbox'][1], object['bbox'][2], object['bbox'][3])
                    objid_list.append(objid)
                    objbbox_list.append(objbbox)
                    objlabel_list.append(objlabel)
                    count += 1
                obj_num = count #len(obj_region)        
            

            inst = dict()
            inst['image_id'] = image_id
            inst['name_verb'] = verb
            # inst['anno_args'] = args
            inst['role_argbbox'] = role_argbbox
            inst['argbboxs'] = arg_bboxes
            inst['name_argtypes'] = arg_type_name
            inst['name_argtypes_all'] = arg_type_name_dict
            inst['desc_verb'] = verb_desc
            inst['desc_argroles'] = arg_role_desc
            inst['name_argroles'] = arg_role_name
            if self.load_object:
                inst['objid'] = objid_list
                inst['objbbox'] = objbbox_list
                inst['objlabel'] = objlabel_list
                inst['obj_num'] = obj_num
            
            self.data.append(inst)

            # break
        
        print('Loaded {} instances from {}'.format(len(self), self.anno_json))

    def get_verb_desc(self, verb):
        if self.prompt == 'def':
            verb_desc = self.ontology_verbs[verb]['def']
        elif self.prompt == 'abstract':
            verb_desc = self.ontology_verbs[verb]['abstract']
        elif self.prompt == 'name':
            verb_desc = verb
        elif self.prompt == 'short':
            verb_desc = 'An image of %s event.' % verb
        else:
            raise RuntimeError("Not defined prompt '{}'".format(self.prompt))
        return verb_desc

    def get_rolename_desc(self, verb, rolename, roletype):
        # print(verb)
        # print(self.ontology_verbs[verb]['roles'])
        if self.prompt == 'def':
            rolename_desc = self.ontology_verbs[verb]['roles'][rolename]['def']
        elif self.prompt == 'abstract':
            rolename_desc = self.ontology_verbs[verb]['roles'][rolename]['framenet']
        elif self.prompt == 'name':
            rolename_desc = 'The %s of %s.' % (rolename, verb)
        elif self.prompt == 'short':
            if roletype is None:
                rolename_desc = 'The object is %s %s.' % (verb, rolename)
            else:
                rolename_desc = 'The %s is %s %s.' % (roletype, verb, rolename)
        return rolename_desc

    def load_candidate_verbs(self):
        verb_list = list(self.ontology_verbs.keys())
        verb_id2str = {idx:verb for idx, verb in enumerate(verb_list)}
        verb_str2id = {verb:idx for idx, verb in verb_id2str.items()}
        verb_vecs = self.tokenize(verb_list).to(self.device)
        print('There are %d verbs in total in this dataset' % len(verb_list))

        return verb_list, verb_vecs, verb_id2str, verb_str2id

    def get_labels(self, class_map_file):
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


    def collate_fn(self, batch): #, preprocess, tokenize):
        
        image_ids = list()
        image_paths = list()
        image_vecs = list()
        role_argbbox = list()
        argbboxs = list()
        # argbbox_vecs = list()

        desc_verbs = list()
        desc_verb_vecs = list()
        desc_argroles = list()
        desc_argrole_vecs = list()

        name_verbs = list()
        name_argroles = list()
        name_argbboxtypes = list()
        name_argbboxtypes_all = list()

        objid_list = list()
        objbbox_list = list()
        objlabel_list = list()
        obj_num = list()
            
            
        for inst in batch:
            image_ids.append(inst['image_id'])
            name_verbs.append(inst['name_verb'])
            desc_verbs.append(inst['desc_verb'])
            desc_verb_vecs.append(self.tokenize(inst['desc_verb']).to(self.device))
            role_argbbox.append(inst['role_argbbox'])
            name_argroles.append(inst['name_argroles'])
            desc_argroles.append(inst['desc_argroles'])
            desc_argrole_vecs.append(self.tokenize(inst['desc_argroles']).to(self.device))
            name_argbboxtypes.append(inst['name_argtypes'])
            name_argbboxtypes_all.append(inst['name_argtypes_all'])

            image_path = os.path.join(self.image_dir, inst['image_id'])
            image_paths.append(image_path)
            image_obj = Image.open(image_path)
            image_vec = self.preprocess(image_obj).to(self.device)
            image_vecs.append(image_vec)

            argbboxs.append(inst['argbboxs'])
            # for argbbox in inst['argbboxs']:
            #     argbbox_obj = image_obj.crop([argbbox[0], argbbox[1], argbbox[2], argbbox[3]])
            #     argbbox_vec = self.preprocess(argbbox_obj).to(self.device)
            #     argbbox_vecs.append(argbbox_vec)

            # TODO: padding
            
            # load object
            if self.load_object:
                objid_list.append(inst['objid'])
                objbbox_list.append(inst['objbbox'])
                objlabel_list.append(inst['objlabel'])
                obj_num.append(inst['obj_num'])
                # # pad the entire image to bbox vectors, in case the bbox is none
                # objid_list.append('0_0_0_0')            
                
        image_vecs = torch.stack(image_vecs, dim=0).to(self.device)
        # argbbox_vecs = torch.stack(argbbox_vecs, dim=0).to(self.device)
        desc_verb_vecs = torch.stack(desc_verb_vecs, dim=0).to(self.device).squeeze(1)
        # TODO: padding
        # desc_argrole_vecs = torch.stack(desc_argrole_vecs, dim=0).to(self.device)

        return Batch(
            image_id=image_ids,
            image_path=image_paths,
            image_vec=image_vecs,
            name_verb=name_verbs,
            desc_verb=desc_verbs,
            desc_verb_vec=desc_verb_vecs,
            argbbox=argbboxs,
            role_argbbox=role_argbbox,
            # argbbox_vec=argbbox_vecs,
            name_argrole=name_argroles,
            desc_argrole=desc_argroles,
            desc_argrole_vec=desc_argrole_vecs,
            name_argtype=name_argbboxtypes,
            name_argtype_all=name_argbboxtypes_all,
            obj_id=objid_list,
            obj_vec=objbbox_list,
            objlabel=objlabel_list,
            objlabel_vec=None,
            obj_num=obj_num,
            
        )


if __name__ == '__main__':
    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    dataset = GSRDataset(
        anno_json='/shared/nas/data/m1/manling2/clip-event/data/gsr/SWiG_jsons/test.json', 
        image_dir='/shared/nas/data/m1/manling2/clip-event/data/gsr/images_512', 
        ontology_json='/shared/nas/data/m1/manling2/clip-event/data/gsr/SWiG_jsons/imsitu_space.json', 
        object_detection='/shared/nas/data/m1/manling2/m2e2/data/mm-event-graph/imSitu/object_detection/det_results_imsitu_oi_1.pkl',
        object_class_map='/shared/nas/data/m1/manling2/m2e2/data/mm-event-graph/object/class-descriptions-boxable.csv',
        object_threshold=0.2, object_topk=40, 
        load_object=True, 
        # anno_json='/home/t-manlingli/clip-event/data/gsr/SWiG_jsons/train.json', 
        # image_dir='/home/t-manlingli/clip-event/data/gsr/images_512', 
        # ontology_json='/home/t-manlingli/clip-event/data/gsr/SWiG_jsons/imsitu_space.json', 
        prompt='name', # def, abstract, name, short
        preprocess=transform, tokenize=tokenize, device=torch.device('cuda')
    )
    loader = DataLoader(
        dataset, batch_size=2, 
        collate_fn=dataset.collate_fn,
        pin_memory=False
    )

    for batch_idx, batch in enumerate(loader):
        # print(batch_idx, batch)
        # image_id = batch.image_id
        # image_vec = batch.image_vec
        # desc_verb_vec = batch.desc_verb_vec

        argbbox = batch.argbbox
        name_argrole = batch.name_argrole
        desc_argrole = batch.desc_argrole
        desc_argrole_vec = batch.desc_argrole_vec
        name_argtype = batch.name_argtype
        name_argtype_all = batch.name_argtype_all

        print('argbbox', argbbox)  ## argbbox [[(0.24384525205158264, 0.072265625, 0.7866354044548651, 0.759765625), None, (0.004689331770222743, 0.1953125, 0.6178194607268465, 0.998046875)], [None, None, None]]
        # print('patch', patch_from_nomalize_bbox(argbbox[0][0], patch_size=7))
        print('name_argrole', name_argrole)  ## name_argrole [['item', 'place', 'agent'], ['item', 'place', 'agent']]
        print('desc_argrole', desc_argrole)  ## desc_argrole [['The item of dialing.', 'The place of dialing.', 'The agent of dialing.'], ['The item of pressing.', 'The place of pressing.', 'The agent of pressing.']]
        print('desc_argrole_vec', desc_argrole_vec)  ## list of tensors
        print('name_argtype', name_argtype)  ## name_argtype [['cellular telephone', 'inside', 'person'], ['picture', 'inside', 'man']]
        print('name_argtype_all', name_argtype_all)

        # print(image_id)
        # print(image_vec.size(), desc_verb_vec.size())
        # # print(argbbox_vec.size(), desc_argrole_vec.size())

        obj_id=batch.obj_id
        obj_vec=batch.obj_vec
        objlabel=batch.objlabel
        objlabel_vec=batch.objlabel_vec
        obj_num=batch.obj_num
        print('obj_id', obj_id)  ## [['54_27_319_213', '116_15_355_192'], ['142_28_405_318', '227_20_430_323', '113_28_403_324', '253_31_436_317']]
        print('obj_vec', obj_vec)  ## [[array([0.11880686, 0.09820018, 0.6954263 , 0.7742871 ], dtype=float32), array([0.2543471 , 0.05796601, 0.77365005, 0.69672996], dtype=float32)], [array([0.3172779 , 0.08865901, 0.90137285, 0.976817  ], dtype=float32), array([0.506293  , 0.06363266, 0.9576754 , 0.9931106 ], dtype=float32), array([0.2516151 , 0.08805192, 0.8960552 , 0.99663615], dtype=float32), array([0.56260705, 0.096701  , 0.97056454, 0.9745097 ], dtype=float32)]]
        print('objlabel', objlabel)  ## [['Mobile phone', 'Mobile phone'], ['Man', 'Person', 'Person', 'Man']]
        print('objlabel_vec', objlabel_vec)  # list of tensors (not matrix)
        print('obj_num', obj_num)

        break