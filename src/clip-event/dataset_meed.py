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
from utils_image import normalize_bbox

instance_fields = [
    'image_id',
    # 'image_path',
    'image_vec',
    'desc_verb',
    'desc_verb_vec',
    # 'name_verb',
    # 'desc_verb',
    # 'desc_verb_vec',
    # 'argbbox',
    # # 'argbbox_vec',
    # 'name_argrole',
    # 'desc_argrole',
    # 'desc_argrole_vec',
    # 'name_argtype', # 'name_argtype_canonical'
]
Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))

batch_fields = [
    'image_id',
    # 'image_path',
    'image_vec',
    'desc_verb',
    'desc_verb_vec',
    # 'name_verb',
    # 'desc_verb',
    # 'desc_verb_vec',
    # 'argbbox',
    # # 'argbbox_vec',
    # 'name_argrole',
    # 'desc_argrole',
    # 'desc_argrole_vec',
    # 'name_argtype',
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


class MEEDDataset(Dataset):
    def __init__(self, anno_json, image_dir, ontology_json, prompt=None, preprocess=None, tokenize=None, device='gpu'):
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

        # # self.vocabs, self.templates = self.load_dict_evt()
        # # {"tattooing": {"framenet": "Create_physical_artwork", "abstract": "AGENT tattooed TARGET with TOOL in PLACE", "def": "to mark the skin with permanent colors and patterns", "order": ["agent", "target", "tool", "place"], "roles": {"tool": {"framenet": "instrument", "def": "The tool used"}, "place": {"framenet": "place", "def": "The location where the tattoo event is happening"}, "target": {"framenet": "representation", "def": "The entity being tattooed"}, "agent": {"framenet": "creator", "def": "The entity doing the tattoo action"}}}, "raining": {"framenet": "Precipitation", "abstract": "it rains in the PLACE", "def": "rain falls.", "order": ["place"], "roles": {"place": {"framenet": "place", "def": "The location where the rain event is happening"}}}
        # self.ontology_verbs = json.load(open(ontology_json))['verbs']  
        # # "n03024882": {"gloss": ["choker", "collar", "dog collar", "neckband"], "def": "necklace that fits tightly around a woman's neck"},
        # self.ontology_nouns = json.load(open(ontology_json))['nouns']
        self.load_data()

        # self.candidate_verbs, self.candidate_verb_vecs, self.verb_id2str, self.verb_str2id = self.load_candidate_verbs()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, item):
        # image reading

        return self.data[item]


    def load_data(self):
        """Load data from file."""

        # load qa pairs
        data_all = json.load(open(self.anno_json))
        for data in data_all:
            image_id = data['image_name']
            verb = data['trigger']['word']
            event = data['event']
            text = data['text']
            # # height = data[image_id]['height']
            # # width = data[image_id]['width']
            # args = data[image_id]['arguments']
            # arg_bboxes = list()
            # arg_role_name = list()
            # arg_role_desc = list()
            # arg_type_name = list()
            # # find the most common noun as obj type
            # arg_type_name_dict = defaultdict(lambda: Counter())
            # for frame in data[image_id]['frames']:
            #     for arg_role in frame:
            #         arg_name_frameid = frame[arg_role]
            #         if len(arg_name_frameid) > 0:
            #             arg_name = self.ontology_nouns[arg_name_frameid]['gloss']
            #             arg_desc = self.ontology_nouns[arg_name_frameid]['def']
            #             arg_type_name_dict[arg_role].update(arg_name)
            # for arg_role in args:
            #     arg_bbox = args[arg_role]
            #     # # (1) have bbox (2) have obj type annotation
            #     # if arg_bbox[0] != -1 and len(arg_type_name_dict[arg_role]) > 0:
            #     #     if arg_bbox[3] == arg_bbox[1]:
            #     #         arg_bbox[3] += 2
            #     #     if arg_bbox[2] == arg_bbox[0]:
            #     #         arg_bbox[2] += 2
            #     #     arg_bboxes.append(arg_bbox)
            #     #     arg_role = arg_role.replace('sources', 'source')
            #     #     arg_role_desc.append(self.get_rolename_desc(verb, arg_role, arg_type_name_dict[arg_role].most_common()[0][0]))
            #     #     arg_role_name.append(arg_role)
            #     #     arg_type_name.append(arg_type_name_dict[arg_role].most_common()[0][0])
            #     arg_role = arg_role.replace('sources', 'source')
            #     if len(arg_type_name_dict[arg_role]) > 0:
            #         arg_type_name.append(arg_type_name_dict[arg_role].most_common()[0][0])
            #         arg_role_desc.append(self.get_rolename_desc(verb, arg_role, arg_type_name_dict[arg_role].most_common()[0][0]))
            #     else:
            #         arg_type_name.append(None)
            #         arg_role_desc.append(self.get_rolename_desc(verb, arg_role, None))
            #     arg_role_name.append(arg_role)
                
            #     if arg_bbox[0] != -1:
            #         # fix the bounding boxes
            #         if arg_bbox[3] == arg_bbox[1]:
            #             arg_bbox[3] += 2
            #         if arg_bbox[2] == arg_bbox[0]:
            #             arg_bbox[2] += 2
            #         arg_bbox = normalize_bbox(arg_bbox, width=width, height=height)
            #         arg_bboxes.append(arg_bbox)
            #     else:
            #         arg_bboxes.append(None)
                    
                

            inst = dict()
            inst['image_id'] = image_id
            # inst['verb'] = verb
            # inst['event'] = event
            # inst['text'] = text
            
            if self.prompt == 'verbprefix':
                inst['desc'] = 'An image of %s' % verb
                self.data.append(inst)
            elif self.prompt == 'eventprefix':
                inst['desc'] =  'An image of %s' % event.split('.')[-1].lower()
                self.data.append(inst)
            elif self.prompt == 'verb':
                inst['desc'] = verb
                self.data.append(inst)
            elif self.prompt == 'event':
                inst['desc'] =  event.split('.')[-1].lower()
                self.data.append(inst)
            elif self.prompt == 'text':
                inst['desc'] =  text[0]
                self.data.append(inst)
                inst2 = dict()
                inst2['image_id'] = image_id
                inst2['desc'] =  text[1]
                self.data.append(inst2)
                inst3 = dict()
                inst3['image_id'] = image_id
                inst3['desc'] =  text[1]
                self.data.append(inst3)
            
            
            # break
        
        print('Loaded {} instances from {}'.format(len(self), self.anno_json))

    def collate_fn(self, batch): #, preprocess, tokenize):
        
        image_ids = list()
        # image_paths = list()
        image_vecs = list()
        # argbboxs = list()
        # # argbbox_vecs = list()

        # desc_verbs = list()
        # desc_verb_vecs = list()
        # desc_argroles = list()
        # desc_argrole_vecs = list()

        # name_verbs = list()
        # name_argroles = list()
        # name_argbboxtypes = list()
        desc_verb_list = list()
        desc_verb_vec_list = list()
            
        for inst in batch:
            image_ids.append(inst['image_id'])
            desc_verb_list.append(inst['desc'])

            image_path = os.path.join(self.image_dir, inst['image_id'])
            # image_paths.append(image_path)
            image_obj = Image.open(image_path)
            image_vec = self.preprocess(image_obj).to(self.device)
            image_vecs.append(image_vec)

            desc_verb_vec = self.tokenize(inst['desc']).to(self.device)
            desc_verb_vec_list.append(desc_verb_vec)

            # argbboxs.append(inst['argbboxs'])
            # for argbbox in inst['argbboxs']:
            #     argbbox_obj = image_obj.crop([argbbox[0], argbbox[1], argbbox[2], argbbox[3]])
            #     argbbox_vec = self.preprocess(argbbox_obj).to(self.device)
            #     argbbox_vecs.append(argbbox_vec)

            # TODO: padding
            
                
        image_vecs = torch.stack(image_vecs, dim=0).to(self.device)
        # argbbox_vecs = torch.stack(argbbox_vecs, dim=0).to(self.device)
        desc_verb_vecs = torch.stack(desc_verb_vec_list, dim=0).to(self.device).squeeze(1)
        # TODO: padding
        # desc_argrole_vecs = torch.stack(desc_argrole_vecs, dim=0).to(self.device)

        return Batch(
            image_id=image_ids,
            # image_path=image_paths,
            image_vec=image_vecs,
            desc_verb=desc_verb_list,
            desc_verb_vec=desc_verb_vecs
            # name_verb=name_verbs,
            # desc_verb=desc_verbs,
            # desc_verb_vec=desc_verb_vecs,
            # argbbox=argbboxs,
            # # argbbox_vec=argbbox_vecs,
            # name_argrole=name_argroles,
            # desc_argrole=desc_argroles,
            # desc_argrole_vec=desc_argrole_vecs,
            # name_argtype=name_argbboxtypes

        )


if __name__ == '__main__':
    transform = Compose([
        Resize(224, interpolation=Image.BICUBIC),
        CenterCrop(224),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    dataset = MEEDDataset(
        anno_json='/shared/nas/data/m1/manling2/clip-event/data/meed/MEED-main/meed.json', 
        image_dir='/shared/nas/data/m1/manling2/clip-event/data/meed/images', 
        ontology_json='/shared/nas/data/m1/manling2/clip-event/data/meed/MEED-main/events.json', 
        # anno_json='/home/t-manlingli/clip-event/data/gsr/SWiG_jsons/train.json', 
        # image_dir='/home/t-manlingli/clip-event/data/gsr/images_512', 
        # ontology_json='/home/t-manlingli/clip-event/data/gsr/SWiG_jsons/imsitu_space.json', 
        prompt='verbprefix', # verb, verbprefix, text, event, eventprefix
        preprocess=transform, tokenize=tokenize, device=torch.device('cuda')
    )
    loader = DataLoader(
        dataset, batch_size=2, 
        collate_fn=dataset.collate_fn,
        pin_memory=False,
        shuffle=True
    )

    for batch_idx, batch in enumerate(loader):
        # print(batch_idx, batch)
        image_id = batch.image_id
        image_vec = batch.image_vec
        desc_verb = batch.desc_verb
        desc_verb_vec = batch.desc_verb_vec
        
        print(image_id, desc_verb)
        print(image_vec.size(), desc_verb_vec.size())
        # # print(argbbox_vec.size(), desc_argrole_vec.size())

        break