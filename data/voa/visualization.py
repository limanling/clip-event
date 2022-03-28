import os
import ujson as json
from collections import defaultdict
import sys
sys.path.append("/shared/nas/data/m1/manling2/aida/util")
from ltf_util import LTF_util

edl_cs = '/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2017/edl/merged.cs'
event_cs = '/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2017/event/event_rewrite.cs'
openie_tab = '/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2017/openie.tab'
ltf_dir = '/shared/nas/data/m1/manling2/mmqa/data/voa_v1_processed/caption_separate/split_year/VOA_EN_NW_2017/ltf'
output_folder = '/shared/nas/data/m1/manling2/clip/data/voa'
visualpath = os.path.join(output_folder, "voa_caption_visualization")

voa_image_caption_file = '/shared/nas/data/m1/manling2/m2e2/data/mm-event-graph/voa/rawdata/voa_img_dataset.json'
voa_image_caption = json.loads(open(voa_image_caption_file).read())

# prefix = 'https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/SeedlingOntology#'

ltf_util = LTF_util(ltf_dir)

enntity_name_mapping = dict()
for line in open(edl_cs).readlines():
    line = line.rstrip('\n')
    tabs = line.split('\t')
    if line.startswith(':Entity'):
        entity_id = tabs[0]
        if 'canonical_mention' == tabs[1]:
            enntity_name_mapping[entity_id] = tabs[2].replace("\"", "")
print(len(enntity_name_mapping))

doc_event = defaultdict(list)
event_dict = defaultdict(lambda : defaultdict(str))
for line in open(event_cs).readlines():
    if line.startswith(':Event'):
        line = line.rstrip('\n')
        tabs = line.split('\t')
        event_id = tabs[0]
        if 'type' in tabs[1]:
            event_dict[event_id]['type'] = tabs[2].split("#")[-1]
        # if 'mention' in tabs[1]:
        elif 'canonical_mention.actual' == tabs[1]:
            offset_str = tabs[3]
            imageid = offset_str[:offset_str.find(':')]
            doc_event[imageid].append(event_id)
            event_dict[event_id]['offset'] = tabs[3] #??? no coreference?
        elif 'mention' not in tabs[1]:
            role = tabs[1].split("#")[-1].replace(".actual", "")
            entity_id = tabs[2]
            event_dict[event_id][role] = '%s:%s' % (entity_id, enntity_name_mapping[entity_id])
print(len(doc_event))

doc_openie = defaultdict(list)
for line in open(openie_tab):
    line = line.rstrip('\n')
    tabs = line.split('\t')
    imageid = tabs[0].split('/')[-1].replace('.rsd.txt', '')
    triples = '(%s, %s, %s)' % (tabs[2], tabs[3], tabs[4])
    doc_openie[imageid].append(triples)
print(len(doc_openie))

# sort by number of events:
doc_sorted = sorted(doc_event.items(), key=lambda x: len(x[1]), reverse=True)

if not os.path.exists(visualpath):
    os.makedirs(visualpath)
record_count = 0
page_limit = 50

for imageid, events in doc_sorted:
    # print(imageid)
    record_count = record_count + 1
    f_html = open(os.path.join(visualpath, 'voa_events_%d.html' % int(record_count / page_limit)), 'a+')
    f_html.write('%s: \n<br>' % (imageid))
    f_html.write('<b>============== IE ================</b>: \n<br>')
    for event_id in events: #doc_event[imageid]:
        offset = event_dict[event_id]['offset']
        # mention = ltf_util.get_str(offset)
        type = event_dict[event_id]['type']
        context = ltf_util.get_context_html(offset)
        f_html.write('<span style="color:red">%s: %s</span>, %s\n<br>' % (event_id, type, context))
        # f_html.write('%s,\t' % (event_id))
        for role in event_dict[event_id]:
            if role != 'offset' and role != 'type':
                f_html.write('[Argument] %s=%s\n<br>' % (role, event_dict[event_id][role]))
    f_html.write('<b>============== OpenIE ================</b>: \n<br>')
    for triple_str in doc_openie[imageid.replace('.', '_')]:
        f_html.write('%s\n<br>' % (triple_str))
    f_html.write('<b>============== Images ================</b>: \n<br>')
    docid = imageid[:imageid.rfind('_')]
    doc_id = list(docid)
    doc_id[14]='.'
    doc_id[17]='.'
    doc_id[20]='.'
    doc_id = ''.join(doc_id)
    for idx in voa_image_caption[doc_id]:
        f_html.write(
            "<img src=\"" + voa_image_caption[doc_id][idx]['url'] + "\" width=\"300\">\n<br>")
    f_html.write('\n<br><br><br>')
    f_html.flush()
    f_html.close()
    # break

head = '''
    <!DOCTYPE html>
    <html>
    <head>
    <title>Page Title</title>
    </head>
    <body>
    '''
tail = '''
    </body>
    </html>
    '''
for html in os.listdir(visualpath):
    if html.endswith('html'):
        html_content = open(os.path.join(visualpath, html)).read()
        html_new = open(os.path.join(visualpath, html), 'w')
        html_new.write('%s\n' % head)
        html_new.write(html_content)
        html_new.write('%s\n' % tail)
        html_new.flush()
        html_new.close()

