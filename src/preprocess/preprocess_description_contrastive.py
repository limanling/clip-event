from collections import defaultdict, Counter
import os
import sys
# import json 
import ujson as json
import random
import re

from ltf_util import parse_offset_str

sys.path.append('/shared/nas/data/m1/manling2/clip-event/src/clip/CLIP-event')
import clip
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def id_normalize(id_raw, suffix):
    return '%s_%s' % (id_raw, suffix) #language.upper()+'_'+id_raw[1:]

def load_cs(input_cs, suffix, doc_ke, entity_info, evt_info, evt_args):
    for line in open(input_cs):
        line = line.rstrip('\n')
        tabs = line.split('\t')

        if line.startswith(':Entity'):
            entity_id = id_normalize(tabs[0], suffix)
            if tabs[1] == 'type':
                entity_type = tabs[2].split('#')[-1]
                if len(tabs) == 4:
                    entity_type_confidence = float(tabs[3])
                elif 'Filler' in entity_id:
                    entity_type_confidence = 1.0
                else:
                    entity_type_confidence = 1.0
                if 'type' not in entity_info[entity_id]:
                    entity_info[entity_id]['type'] = dict()
                entity_info[entity_id]['type'][entity_type] = entity_type_confidence
            elif 'canonical_mention' in tabs[1]:
                offset = tabs[3]
                mention_str = tabs[2][1:-1]
                docid, start, end = parse_offset_str(offset)
                entity_info[entity_id]['confidence'] = float(tabs[4])
                entity_info[entity_id]['canonical_mention'] = mention_str #offset
                if entity_id not in doc_ke[docid]['entity']:
                    doc_ke[docid]['entity'].append(entity_id)
            elif 'mention' in tabs[1]:
                offset = tabs[3]
                mention_type = tabs[1].replace(".actual", "")
                mention_confidence = float(tabs[4])
                mention_str = tabs[2][1:-1]
                if 'mention' not in entity_info[entity_id]:
                    entity_info[entity_id]['mention'] = dict()
                entity_info[entity_id]['mention'][offset] = (mention_type, mention_str) 
            elif 'link' in tabs[1]:
                link_target = tabs[2]
                if 'link' not in entity_info[entity_id]:
                    entity_info[entity_id]['link'] = dict()
                if len(tabs) > 3:
                    link_confidence = tabs[3]
                    entity_info[entity_id]['link'][link_target] = link_confidence
                else:
                    entity_info[entity_id]['link'][link_target] = 1.0
        if line.startswith(':Event') or line.startswith(':Relation'):
            ke_type = line[1:line.find('_')].lower()
            evt_id = id_normalize(tabs[0], suffix)
            if tabs[1] == 'type':
                evt_info[evt_id]['type'] = tabs[2].split('#')[-1].strip()
            elif 'canonical_mention' in tabs[1]:
                offset = tabs[3]
                mention_str = tabs[2][1:-1]
                docid, start, end = parse_offset_str(offset)
                evt_info[evt_id]['confidence'] = float(tabs[4])
                evt_info[evt_id]['canonical_mention'] = mention_str #offset
                if evt_id not in doc_ke[docid][ke_type]:
                    doc_ke[docid][ke_type].append(evt_id)
            elif 'mention' in tabs[1]: 
                offset = tabs[3]
                mention_type = tabs[1].replace(".actual", "")
                mention_confidence = float(tabs[4])
                mention_str = tabs[2][1:-1]
                if 'mention' not in evt_info[evt_id]:
                    evt_info[evt_id]['mention'] = dict()
                evt_info[evt_id]['mention'][offset] = (mention_type, mention_str)   
            elif 'Entity' in tabs[2] or 'Filler' in tabs[2]:
                role = tabs[1].split('#')[-1].replace(".actual", "") # no other label than ".actual" for now
                arg_id = id_normalize(tabs[2], suffix)
                # print(role, arg_id, arg_id in doc_ke[docid]['entity'], doc_ke[docid])
                # if arg_id not in doc_ke[docid]['entity']:
                #     continue
                arg_offset = tabs[3]
                arg_confidence = float(tabs[4])
                if arg_id not in evt_args[evt_id][role]:
                    arg_mention_type, arg_mention_str = entity_info[arg_id]['mention'][arg_offset]
                    arg_mention_canonical = entity_info[arg_id]['canonical_mention']
                    evt_args[evt_id][role][arg_id] = (arg_offset, arg_mention_type, arg_mention_str, arg_mention_canonical) 
                # print(evt_args[evt_id][role])
            # elif tabs[1].startswith('t') and len(tabs[1]) == 2:
            #     t_num = tabs[1]
            #     date = tabs[2]
            #     # for event_id, t_num, date in four_tuples:
            #     num = int(t_num[1:]) - 1
            #     if "inf" not in date:
            #         date = convert_data_gdate(date)
            #     else:
            #         if num < 3:
            #             date = convert_data_gdate("_9999-01-01")
            #         else:
            #             date = convert_data_gdate("9999-12-31")
            #     if 'time' not in evt_info[evt_id]: 
            #         evt_info[evt_id]['time'] = [None, None, None, None]
            #     evt_info[evt_id]['time'][num] = date
                
    # return doc_ke, entity_info, evt_info, evt_args


def get_image_clippred(doc_id, image_dir_dict, model, preprocess, text, device):
    try:
        image_clip_file = os.path.join('/shared/nas/data/m1/manling2/clip-event/data/voa/clip', doc_id+'.json')
        if os.path.exists(image_clip_file):
            image_clip_dict = json.load(open(image_clip_file))
            scores = image_clip_dict['scores']
            pred_idx = image_clip_dict['pred_idx']
            probs = image_clip_dict['probs']
        else:
            image_dir_prefix = doc_id[:14]
            image_path = os.path.join(image_dir_dict[image_dir_prefix], doc_id+'.jpg')
            image_vec = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image_vec, text)
                probs = logits_per_image.softmax(dim=-1)
                scores, pred_idx = torch.max(probs, dim=-1)
                scores = scores.item()
                pred_idx = pred_idx.item()
                probs = probs.tolist()
                # print(scores, pred_idx)
                # json.dump({doc_id: {'scores': scores, 'pred_idx': pred_idx, 'probs':probs}}, open(os.path.join('/shared/nas/data/m1/manling2/clip-event/src/preprocess/voa', doc_id+'.json'), 'w'))
                json.dump({'scores': scores, 'pred_idx': pred_idx, 'probs':probs}, open(image_clip_file, 'w'))
    except:
        print(image_path)
        print(sys.exc_info())
        scores = 0
        pred_idx = -1
        probs = list()
    return scores, pred_idx, probs

def select_postive_event(doc_id, doc_ke, entity_info, evt_info, evt_args, image_dir_dict, add_clip_sim=True, model=None, preprocess=None, text=None, device='cpu', id2str=None, str2id=None, merge_args=True):
    if len(doc_ke[doc_id]['event']) == 0:
        return None, None
    
    # only one event, directly return this event
    if len(doc_ke[doc_id]['event']) == 1:
        for event_id in doc_ke[doc_id]['event']:
            return event_id, evt_args[event_id]

    if add_clip_sim:
        scores, pred_idx, probs = get_image_clippred(doc_id, image_dir_dict, model, preprocess, text, device)

    # get counters
    evt_type_counter = Counter()
    arg_num_all = 0
    for event_id in doc_ke[doc_id]['event']:
        event_type = evt_info[event_id]['type']
        evt_type_counter[event_type] += 1
        arg_num_all += len(evt_args[event_id])
    # evt_type_top1 = evt_type_counter.most_common()
    
    # step 1. Run event coreference system [Lai et al, 2021]
    # step 2. Most frequent event type
    # step 3. The largest number of extracted arguments
    # step 4. CLIP similarity of <trigger word, image>

    event_ranker = defaultdict(float)
    for event_id in doc_ke[doc_id]['event']:
        # event type
        event_ranker[event_id] += evt_type_counter[evt_info[event_id]['type']] / float(len(doc_ke[doc_id]['event']))
        # arg num
        if arg_num_all > 0:
            event_ranker[event_id] += len(evt_args[event_id]) #/ float(arg_num_all)
        # clip similarity
        if add_clip_sim:
            # print(id2str)
            # print(id2str[pred_idx], evt_info[event_id]['type'], scores, event_ranker[event_id])
            if pred_idx != -1:
                pred_type_str = id2str[pred_idx]
                if (pred_type_str == evt_info[event_id]['type']):
                    event_ranker[event_id] += scores * 10
                elif pred_type_str.split('.')[0] == evt_info[event_id]['type'].split('.')[0]:
                    # parent is the same
                    event_ranker[event_id] += scores * 5
                else:
                    event_ranker[event_id] -= scores * 10
            else:
                # no pred types from image
                pass #event_ranker[event_id] -= scores * 10
    event_ranked = sorted(event_ranker.items(), key=lambda x:x[1], reverse=True)
    event_ranked_id = event_ranked[0][0]
    # print('selected: ', evt_info[event_ranked_id]['type'])

    event_ranked_args = evt_args[event_ranked_id]
    if merge_args:
        event_ranked_type = evt_info[event_ranked_id]['type']
        for event_id in doc_ke[doc_id]['event']:
            if event_ranked_type == evt_info[event_id]['type']:
                for role in evt_args[event_id]:
                    for arg_id in evt_args[event_id][role]:
                        if role not in event_ranked_args or arg_id not in event_ranked_args[role]:
                            event_ranked_args[role][arg_id] = evt_args[event_id][role][arg_id]
    return event_ranked_id, event_ranked_args

def select_postive_event_all(doc_salient_event, doc_ke, entity_info, evt_info, evt_args, image_dir_dict, add_clip_sim=True, model=None, preprocess=None, text=None, device='cpu', id2str=None, str2id=None, merge_args=True):
    none_event_num = 0
    event_num = 0
    for doc_id in doc_ke:
        event_ranked_id, event_ranked_args = select_postive_event(doc_id, doc_ke, entity_info, evt_info, evt_args, image_dir_dict, add_clip_sim=add_clip_sim, model=model, preprocess=preprocess, text=text, device=device, id2str=id2str, str2id=str2id, merge_args=merge_args)
        if event_ranked_id is not None:
            # doc_salient_event[doc_id] = event_selected
            doc_salient_event[doc_id]['event_id'] = event_ranked_id
            doc_salient_event[doc_id]['event_type'] = evt_info[event_ranked_id]['type']
            doc_salient_event[doc_id]['event_trigger'] = evt_info[event_ranked_id]['canonical_mention']
            doc_salient_event[doc_id]['event_args'] = event_ranked_args
            # doc_salient_event[doc_id]['caption'] = 
            event_num += 1
        else:
            none_event_num += 1
    print("caption_no_event:", none_event_num, "caption_event:", event_num)

def preprocess_event_selection(output_dir, add_clip_sim=False, model=None, preprocess=None, text=None, device='cpu', id2str=None, str2id=None):
    cs_input_list = {
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2009/edl/merged.cs": "2009",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2009/event/event_rewrite.cs": "2009",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2010/edl/merged.cs": "2010",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2010/event/event_rewrite.cs": "2010",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2011/edl/merged.cs": "2011",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2011/event/event_rewrite.cs": "2011",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2012/edl/merged.cs": "2012",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2012/event/event_rewrite.cs": "2012",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2013/edl/merged.cs": "2013",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2013/event/event_rewrite.cs": "2013",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2014/edl/merged.cs": "2014",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2014/event/event_rewrite.cs": "2014",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2015/edl/merged.cs": "2015",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2015/event/event_rewrite.cs": "2015",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2016/edl/merged.cs": "2016",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2016/event/event_rewrite.cs": "2016",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2017/edl/merged.cs": "2017",
        "/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2017/event/event_rewrite.cs": "2017"
    }
    
    image_dir_dict = {
        "VOA_EN_NW_2017":"/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2017/vision/data/jpg/jpg", 
        "VOA_EN_NW_2016":"/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2016/vision/data/jpg/jpg", 
        "VOA_EN_NW_2015":"/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2015/vision/data/jpg/jpg", 
        "VOA_EN_NW_2014":"/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2014/vision/data/jpg/jpg", 
        "VOA_EN_NW_2013":"/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2013/vision/data/jpg/jpg", 
        "VOA_EN_NW_2012":"/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2012/vision/data/jpg/jpg", 
        "VOA_EN_NW_2011":"/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2011/vision/data/jpg/jpg", 
        "VOA_EN_NW_2010":"/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2010/vision/data/jpg/jpg", 
        "VOA_EN_NW_2009":"/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2009/vision/data/jpg/jpg"
    }

    doc_ke = defaultdict(lambda: defaultdict(list))
    entity_info = defaultdict(lambda : defaultdict())
    evt_info = defaultdict(lambda : defaultdict())
    evt_args = defaultdict(lambda : defaultdict(lambda : defaultdict()))
    doc_salient_event = defaultdict(lambda : defaultdict())

    for cs_input in cs_input_list:
        suffix = cs_input_list[cs_input]
        load_cs(cs_input, suffix, doc_ke, entity_info, evt_info, evt_args)

    # # test event selection
    # # doc_id = 'VOA_EN_NW_2009_11_21_415847_4' # size=1
    # doc_id = 'VOA_EN_NW_2009_12_13_416444_3'
    # event_selected = select_postive_event(doc_id, doc_ke, entity_info, evt_info, evt_args, add_clip_sim=add_clip_sim)
    # print(event_selected)
    select_postive_event_all(doc_salient_event, doc_ke, entity_info, evt_info, evt_args, image_dir_dict, add_clip_sim=add_clip_sim, model=model, preprocess=preprocess, text=text, device=device,id2str=id2str, str2id=str2id, merge_args=merge_args)

    json.dump(doc_ke, open(os.path.join(output_dir, 'doc_ke.json'), 'w'), indent=4) #, default=set_default)
    json.dump(entity_info, open(os.path.join(output_dir, 'entity_info.json'), 'w'), indent=4) #, default=set_default)
    json.dump(evt_info, open(os.path.join(output_dir, 'evt_info.json'), 'w'), indent=4) #, default=set_default)
    json.dump(evt_args, open(os.path.join(output_dir, 'evt_args.json'), 'w'), indent=4) #, default=set_default)
    json.dump(doc_salient_event, open(os.path.join(output_dir, 'doc_salient_event_%s_merge%s.json' % (add_clip_sim, merge_args)), 'w'), indent=4) #, default=set_default)

def preprocess_caption(image_caption_json_clean):
    # image_caption_json = '/home/t-manlingli/clip-event/data/voa/image_caption_mapping.json' 
    image_caption_json = '/shared/nas/data/m1/manling2/public_html/voa_data/source_data/image_caption_mapping.json'
    data = json.load(open(image_caption_json))

    image_caption_clean = dict()
    for doc_id in data:
        for image_idx in data[doc_id]:
            image_id = '%s_%s' % (doc_id, image_idx)
            image_id = image_id.replace('.', '_')
            caption = data[doc_id][image_idx]['cap'].replace('FILE - ', '')
            image_caption_clean[image_id] = caption
    json.dump(image_caption_clean, open(image_caption_json_clean, 'w'), indent=4)


def short_template_type(event_type):
    event_type_str_list = re.findall('[A-Z][^A-Z]*', event_type.split('.')[-1])
    if len(event_type_str_list) > 0:
        event_type_str = ' '.join(event_type_str_list).lower()
    else:
        event_type_str = event_type.split('.')[-1]
    template_short = "An image of %s event. " % event_type_str
    return template_short

def short_template_role(role, args_str):
    template_short = "The %s are %s. " % (role.lower(), args_str.lower())
    return template_short

def edit_type(caption, trigger_word, event_type_neg):
    trigger_neg = ' '.join(re.findall('[A-Z][^A-Z]*', event_type_neg.split('.')[-1])).lower()
    return caption.replace(trigger_word, trigger_neg)

def neg_template(positive_option, negative_option, template_file, doc_salient_event, doc_caption,
                output_posneg, neg_num=1, use_rolename=True, sample_neg_arg=False):
    template_dict = json.load(open(template_file))

    # parent type -> event type belonging to the parent type
    event_type_pos_set = defaultdict(set)
    # parent type -> neg candidate type
    event_type_neg_set = defaultdict(list)
    for doc_id in doc_salient_event:
        event_type = doc_salient_event[doc_id]['event_type']
        event_type_parent = event_type.split('.')[0]
        event_type_pos_set[event_type_parent].add(event_type)
    for event_type_parent in event_type_pos_set:
        for event_type_parent_neg in event_type_pos_set:
            if event_type_parent_neg != event_type_parent:
                event_type_neg_set[event_type_parent_neg].extend(event_type_pos_set[event_type_parent])

    # generate template based description
    posneg_descriptions = defaultdict(lambda : defaultdict(list))
    for doc_id in doc_salient_event:
        event_id = doc_salient_event[doc_id]['event_id']
        event_trigger = doc_salient_event[doc_id]['event_trigger']
        event_type = doc_salient_event[doc_id]['event_type']
        event_args = doc_salient_event[doc_id]['event_args']
        caption = doc_caption[doc_id]

        if len(event_args) < 0:
            # only include those with args
            continue

        # load positive template
        template = template_dict[event_type]['template']
        template_short = short_template_type(event_type)
        
        # sample negative of event level
        event_type_parent = event_type.split('.')[0]
        event_type_neg_candidate = event_type_neg_set[event_type_parent]
        event_type_neg_sample_types = random.sample(event_type_neg_candidate, neg_num)
        # load negative template
        event_type_neg_samples = [template_dict[_]['template'] for _ in event_type_neg_sample_types]
        event_type_neg_samples_short = [short_template_type(_) for _ in event_type_neg_sample_types]
        event_type_neg_samples_caption = [edit_type(caption, event_trigger, _) for _ in event_type_neg_sample_types]

        # fill template by args
        filled_template = template
        filled_template_neg = template
        filled_template_short = template_short
        filled_template_neg_short = template_short
        filled_template_neg_caption = caption
        filled_arg_roles = dict()
        # fill correct args
        for role in event_args:
            args = event_args[role]
            role = role.split('_')[-1].replace('Prosecutor', 'Adjudicator')
            role_idx = template_dict[event_type]['roles'].index(role)
            role_idxstr = '<arg%d>' % (role_idx + 1)
            filled_args = set()
            for arg_id in args:
                offset, mention_type, mention, canonical_mention = args[arg_id]
                filled_args.add(canonical_mention)
            filled_args = ' and '.join(filled_args)
            filled_arg_roles[role] = filled_args
            # # fill the arguments for the positive event template
            filled_template = filled_template.replace(role_idxstr, filled_args)
            filled_template_short = filled_template_short + short_template_role(role, filled_args)
            # fill the arguments for the negative event template, according to the argument index
            event_type_neg_samples = [event_type_neg_sample.replace(role_idxstr, filled_args) for event_type_neg_sample in event_type_neg_samples]
            event_type_neg_samples_short = [event_type_neg_sample_short + short_template_role(role, filled_args) for event_type_neg_sample_short in event_type_neg_samples_short]
            # event_type_neg_samples_caption = event_type_neg_samples_caption
        if 'caption' in positive_option:
            posneg_descriptions[doc_id]['pos'].append(doc_caption[doc_id])
        if 'template' in positive_option:
            posneg_descriptions[doc_id]['pos'].append(filled_template)
        if 'short' in positive_option:
            posneg_descriptions[doc_id]['pos'].append(filled_template_short)
        if 'shortverb' in positive_option:
            posneg_descriptions[doc_id]['pos'].append(template_short)
        if 'template' in negative_option:
            posneg_descriptions[doc_id]['neg_event'].extend(event_type_neg_samples)
        if 'short' in negative_option:
            posneg_descriptions[doc_id]['neg_event'].extend(event_type_neg_samples_short)
        if 'caption' in negative_option:
            posneg_descriptions[doc_id]['neg_event'].extend(event_type_neg_samples_caption)
        if 'shortverb' in negative_option:
            posneg_descriptions[doc_id]['neg_event'].extend([short_template_type(_) for _ in event_type_neg_sample_types])
            

        # generate negative template of argument level
        candidate_arg_roles = set()
        for role in template_dict[event_type]['roles']:
            # role = role.split('_')[-1].replace('Prosecutor', 'Adjudicator')
            if role not in filled_arg_roles:
                candidate_arg_roles.add(role)

        if sample_neg_arg:
            if len(candidate_arg_roles) > 0:
                # sampling rather than switching
                arg_neg_samples = random.sample(list(candidate_arg_roles), neg_num)
                # sample a negative argument
                for arg_neg_sample in arg_neg_samples:
                    role_idx_neg = template_dict[event_type]['roles'].index(arg_neg_sample)
                    role_idxstr_neg = '<arg%d>' % (role_idx_neg + 1)
                    # only replace one argument for each neg description
                    filled_template_neg = filled_template_neg.replace(role_idxstr_neg, filled_args)
            else:
                # switch
                arg_neg_samples = random.sample(list(filled_arg_roles), neg_num)
                # sample a negative argument
                for arg_neg_sample in arg_neg_samples:
                    role_idx_neg = template_dict[event_type]['roles'].index(arg_neg_sample)
                    role_idxstr_neg = '<arg%d>' % (role_idx_neg + 1)
                    # only replace one argument for each neg description
                    filled_template_neg = filled_template_neg.replace(role_idxstr_neg, filled_args)
                    # switch the neg role
                    filled_args_neg = filled_arg_roles[arg_neg_sample]
                    filled_template_neg = filled_template_neg.replace(role_idxstr, filled_args_neg)
        else:
            candidate_arg_roles = list(template_dict[event_type]['roles'])
            for role_filled in filled_arg_roles:
                role_neg = random.sample(candidate_arg_roles, 1)[0]
                if role_neg == role_filled:
                    role_neg = random.sample(candidate_arg_roles, 1)[0]
                # remove the sampled neg role from candidate
                candidate_arg_roles.remove(role_neg)
                # fill negative
                filled_args_filled = filled_arg_roles[role_filled]
                role_idx_neg = template_dict[event_type]['roles'].index(role_neg)
                role_idxstr_neg = '<arg%d>' % (role_idx_neg + 1)
                filled_template_neg = filled_template_neg.replace(role_idxstr_neg, filled_args_filled)
                filled_template_neg_short = filled_template_neg_short + short_template_role(role_neg, filled_args_filled)
                if role_neg in filled_arg_roles:
                    filled_args_filled_neg = filled_arg_roles[role_neg]
                    filled_template_neg_caption = filled_template_neg_caption.replace(filled_args_filled, filled_args_filled_neg)
                else:
                    filled_template_neg_caption = filled_template_neg_caption.replace(filled_args_filled, role_neg.lower())

        
        # fill other arguments in the neg_argument by the correct arguments
        for role_ in event_args:
            args_ = event_args[role_]
            role_ = role_.split('_')[-1].replace('Prosecutor', 'Adjudicator')
            role_idx_ = template_dict[event_type]['roles'].index(role_)
            role_idxstr_ = '<arg%d>' % (role_idx_ + 1)
            if role_idxstr == role_idxstr_:
                # this one is used to fill the negative, so ignore
                continue
            filled_args_ = set()
            for arg_id_ in args_:
                offset, mention_type, mention, canonical_mention_ = args_[arg_id_]
                filled_args_.add(canonical_mention_)
            filled_args_ = ' and '.join(filled_args_)
            filled_template_neg = filled_template_neg.replace(role_idxstr_, filled_args_)
            # filled_template_neg_short = filled_template_neg_short + short_template_role(role_, filled_args_)
            # filled_template_neg_caption
        if 'template' in negative_option:
            posneg_descriptions[doc_id]['neg_argument'].append(filled_template_neg)
        if 'short' in negative_option:
            posneg_descriptions[doc_id]['neg_argument'].append(filled_template_neg_short)
        if 'caption' in negative_option:
            posneg_descriptions[doc_id]['neg_argument'].append(filled_template_neg_caption)
        if 'shortverb' in negative_option:
            pass
        
        # replace all <argx> as the role name
        if use_rolename:
            for template_type in posneg_descriptions[doc_id]:
                posneg_descriptions[doc_id][template_type] = [rename_args(template_, template_dict[event_type]['roles']) for template_ in posneg_descriptions[doc_id][template_type]]

    json.dump(posneg_descriptions, open(output_posneg, 'w'), indent=4) #, default=set_default)

def rename_args(template_, template_dict_event_type):
    for role_idx_, role_name_ in enumerate(template_dict_event_type):
        template_ = template_.replace('<arg%d>' % (role_idx_ + 1), role_name_.lower())
    return template_


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def load_clip_sim(clip_ckpt_path):
    ## load openai model  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_ckpt_path, device=device)
    model.eval()

    # /shared/nas/data/m1/manling2/clip-event/src/preprocess/voa/preprocess_description_typename.py
    dict_json = '/shared/nas/data/m1/manling2/clip-event/src/preprocess/voa/conf/ontology_oneie.json'
    id2str = dict()
    str2id = dict()
    event_desc_list = list()
    idx = 0
    data = json.load(open(dict_json))
    for event_type in data:
        id2str[idx] = event_type
        str2id[event_type] = idx
        idx += 1
        event_desc = data[event_type]['desc_auto_name']
        # if len(event_type.split('.')) == 3:
        #     if event_type.split('.')[1] == event_type.split('.')[2]:
        #         event_type_str = event_type.split('.')[2]
        #     else:
        #         event_type_str = event_type.split('.')[2] + 'In' + event_type.split('.')[1] # ' '.join(event_type.split('.')[1:])
        # else:
        #     event_type_str = event_type.split('.')[-1]
        # event_type_str = re.findall('[A-Z][^A-Z]*', event_type_str)
        # event_desc = 'An image of %s.' % (' '.join(event_type_str).lower())
        # print(event_type, event_desc)
        event_desc_list.append(event_desc)
        # data[event_type]['desc_auto_name'] = event_desc
    text = clip.tokenize(event_desc_list).to(device)

    json.dump(data, open(dict_json, 'w'), indent=4)

    return model, preprocess, text, device, id2str, str2id



if __name__ == '__main__':
    add_clip_sim = True
    merge_args = True
    use_rolename = True
    output_dir = '/shared/nas/data/m1/manling2/clip-event/data/voa/ie' #'/home/t-manlingli/clip-event/data/voa/ie' #
    image_caption_json_clean = os.path.join(output_dir, 'image_caption_clean.json')
    clip_ckpt_path = '/shared/nas/data/m1/manling2/clip-event/checkpoint/ViT-B-32.pt'

    # if add_clip_sim:
    #     model, preprocess, text, device, id2str, str2id = load_clip_sim(clip_ckpt_path)
    #     preprocess_event_selection(output_dir, add_clip_sim=add_clip_sim, model=model, preprocess=preprocess, text=text, device=device, id2str=id2str, str2id=str2id)
    # else:
    #     preprocess_event_selection(output_dir, add_clip_sim=add_clip_sim)
    
    ################## generate constractive description ################
    
    preprocess_caption(image_caption_json_clean)

    # positive_option = ['template', 'caption']
    # positive_option = ['short']
    # positive_option = ['shortverb']
    # positive_option = ['caption']
    positive_option = ['template']

    # negative_option = ['short'] #'template-short'
    negative_option = ['template']
    # negative_option = ['caption'] # caption-edit
    # negative_option = ['shortverb']

    # template_dict = '/home/t-manlingli/clip-event/src/preprocess/voa/conf/ontology_oneie.json'
    template_dict = '/shared/nas/data/m1/manling2/clip-event/src/preprocess/voa/conf/ontology_oneie.json'
    output_posneg = os.path.join(output_dir, 'descriptions_%s_%s.json' % (''.join(positive_option), ''.join(negative_option)))
    
    doc_salient_event = json.load(open(os.path.join(output_dir, 'doc_salient_event_%s_merge%s.json' % (add_clip_sim, merge_args))))
    doc_caption = json.load(open(image_caption_json_clean))
    neg_template(positive_option, negative_option, template_dict, doc_salient_event, doc_caption, output_posneg, neg_num=1, use_rolename=use_rolename)