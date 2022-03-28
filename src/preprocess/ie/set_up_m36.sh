#!/usr/bin/env bash

kb_dir=$1  #/scratch/xiaoman6/tmp/edl_data/kb/LDC2019E43_AIDA_Phase_1_Evaluation_Reference_Knowledge_Base/data

docker pull mongo
docker pull panx27/edl
docker pull limanling/uiuc_ie_m36
docker pull dylandilu/event_coreference_xdoc
# docker pull wangqy96/aida_nominal_coreference_en
docker pull panx27/data-processor
docker pull limanling/aida-tools
# docker pull dylandilu/filler
docker pull dylandilu/chuck_coreference
# docker pull lifuhuang/aida_event_lf
docker pull limteng/oneie_aida_m36
docker pull wenhycs/uiuc_event_time
# docker pull laituan245/aida_attrs_filter
docker pull panx27/aida20_mention
docker pull laituan245/spanbert_entity_coref
docker pull laituan245/spanbert_coref
docker pull laituan245/es_event_coref
docker pull laituan245/es_spanbert_entity_coref


if [ -d "${PWD}/system/aida_edl" ]
then
    echo "KB for linking is already in "${PWD}"/system/aida_edl"
else
    docker run --rm -v `pwd`:`pwd` -w `pwd` -i limanling/uiuc_ie_m36 mkdir -p ${PWD}/system/aida_edl
    docker run -v ${PWD}/system/aida_edl:/data panx27/data-processor wget http://159.89.180.81/demo/resources/edl_data.tar.gz -P /data
    docker run -v ${PWD}/system/aida_edl:/data panx27/data-processor tar zxvf /data/edl_data.tar.gz -C /data
fi

docker run -d --rm -v ${PWD}/system/aida_edl/edl_data/db:/data/db --name db mongo

if [ -d "${kb_dir}" ]
then
    docker run --rm --link db:mongo -v ${kb_dir}:/data panx27/edl python ./projs/docker_aida19/kb/import_kb.py /data/entities.tab
    docker run --rm --link db:mongo -v ${kb_dir}:/data panx27/edl python ./projs/docker_aida19/kb/import_mentions.py /data/entities.tab
fi

docker run -d -i --rm --name uiuc_ie_m36 -w /entity_api -p 5500:5500 --name aida_entity --gpus all limanling/uiuc_ie_m36 \
    /opt/conda/envs/aida_entity/bin/python \
    /entity_api/entity_api/app.py --eval m36

# docker run -d -i --rm -w /event_coreference_xdoc -p 6001:6001 --name event_coreference_en dylandilu/event_coreference_xdoc python aida_event_coreference_backen_eng.py
docker run -d -i --rm -w /event_coreference_xdoc -p 6301:6301 --name event_coreference_ru dylandilu/event_coreference_xdoc python aida_event_coreference_backen_rus.py
# docker run -d -i --rm -w /event_coreference_xdoc -p 6201:6201 --name event_coreference_uk dylandilu/event_coreference_xdoc python aida_event_coreference_backen_ukr.py
docker run -d -i --rm -w /event_coreference_xdoc -p 6101:6101 --name event_coreference_es dylandilu/event_coreference_xdoc python aida_event_coreference_backen_spa.py