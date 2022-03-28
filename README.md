# Improving visual event and argument role understanding with contrastive image-language pretraining

## Table of Contents
=================
  * [Overview](#overview)
  * [Requirements](#requirements)
      * [Docker](#docker)
      * [Installing from scratch](#installing)
  * [Data](#data)
      * [VOA training data](#)
      * [Evaluation data](#)
  * [Code](#code)
      * [Code structure](#)
      * [Our method CLIP-event (core code)](#)
  * [Model](#model)
      * [Best performed models](#)
  * [Preprocessing](#preprocessing)
      * [Text IE scripts](#)
      * [Text OpenIE scripts](#)
      * [Vision IE scripts](#)
      * [IE visualizations](#)
      * [Negative description sample generation](#)
  * [Training](#training)
      * [Configuration file](#)
      * [Training on dev box](#)
      * [Training on ITP](#)
  * [Testing](#testing)
  * [OtherDocuments](#otherdocuments)

## Overview
Real-world multimedia applications require image-language models to understand multiple levels of alignments such as verbs, objects, as well as semantic structures. However, existing image-language pretraining models focus on the understanding of images or objects, ignoring the verb semantics and structures. Also, they heavily rely on fine-tuning, while real-world applications require the ability to handle open vocabulary verbs. In this work, we introduce a contrastive learning framework to enforce V+L models to understand events and their argument roles by taking the advantage of text information extraction tools to generate hard negative descriptions. To enforce the model to understand event structures, we design a graph alignment loss via optimal transport, and also augment transformers with local attention heads over events and arguments. To evaluate the model's ability to handle open vocabulary verbs, our experiments are conducted in an unsupervised setting, showing that our model can achieve considerable improvements on a variety of tasks such as multimedia event extraction, grounded situation recognition, visual commonsense reasoning, etc.

## Requirements

### Docker
The docker for V100 GPUs is `limanling/clip-event:v100` and for A100 GPUs is `limanling/clip-event:a100`.

### Installing from scratch
You can also choose to set up the environment from scratch. The code is based on [CLIP](https://github.com/openai/CLIP), and additional dependencies are detailed in `docker/v100-docker/requirements.txt`. 
```
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install git+https://github.com/openai/CLIP.git
pip install -r docker/v100-docker/requirements.txt
```
Please replace cudatoolkit=11.0 above with the appropriate CUDA version on your machine or cpuonly when installing on a machine without a GPU.

## Data

### VOA raw data and extraction result
Please find the VOA data in sdrgprmblob01scus:`t-manlingli/voa`. The data is grouped by the publishing year, resulting in 9 directories including `VOA_EN_NW_2009`, `VOA_EN_NW_2010`, `VOA_EN_NW_2011`, `VOA_EN_NW_2012`, `VOA_EN_NW_2013`, `VOA_EN_NW_2014`, `VOA_EN_NW_2015`, `VOA_EN_NW_2016`, `VOA_EN_NW_2017`. We also provide a small sample dataset with 50 documents (98 images) for quick testing, named `VOA_EN_NW_2017_sample50`, and the testing captions are in `t-manlingli/voa/small/image_caption_mapping_small.json`. 

The file structure for each directory is the same. Take `VOA_EN_NW_2009` as an example,
```
---- VOA_EN_NW_2009
-------- cu_objdet_results
------------ det_results_merged_34a.pkl  # object detection result
-------- edl
------------ merged.cs  # entity extraction and entity linking/coreference result
-------- event
------------ event_rewrite.cs  # event extraction result
-------- ltf
------------ *.ltf.xml  # tokenized captions
-------- rsd
------------ *.rsd.txt  # raw captions
-------- relation
------------ en.rel.cs  # relation extraction result
-------- vision
------------ data/jpg/jpg  # VOA images
------------ docs/image_caption_mapping.json  # the caption information and raw URL for each image
------------ excluded  # the images that being filtered out
```
The `*.cs` files are in Cold Start format, and please find the detailed format instructions in [Cold Start Format](https://docs.google.com/document/d/1DS2TX2syeJ8Xzy0fbhZe0YaqsU5NCqOTedtXZbLczuM/edit?usp=sharing). 

The positive and negative descriptions are under `ie` directory:
```
---- ie
-------- descriptions_caption_caption.json  # postive and negative descriptions using caption editing
-------- descriptions_short_short.json  # postive and negative descriptions using short templates
-------- descriptions_shortverb_shortverb.json  # postive and negative descriptions using short templates (only using verbs, removing argument templates)
-------- descriptions_template_template.json  # postive and negative descriptions using long templates
-------- doc_ke.json  # text IE results for all captions
-------- doc_salient_event_False_mergeTrue.json  # the salient events extracted from each caption
-------- entity_info.json  # entities extracted from captions
-------- evt_info.json  # events extracted from captions
-------- evt_args.json  # event arguments extracted from captions
```

### Testing data

- Multimedia event extraction (M2E2): 

Please find images in `t-manlingli/m2e2/image`, and the text event ground truth in `t-manlingli/m2e2/article_0816_filter.json`, vision event ground truth in `t-manlingli/m2e2/image_event.json`.

- Grounded situation recognition (GSR): 

Please find images in `t-manlingli/gsr/images_512`, and the annotation data in `t-manlingli/gsr/SWiG_jsons`. The verb ontology is in `t-manlingli/gsr/SWiG_jsons/imsitu_space.json`. 

- Visual commonsense reasoning (VCR)

Please find images in `t-manlingli/vcr/images/vcr1images`, and the annotation data in `t-manlingli/vcr/{train,val,test}.jsonl`. 

- Visual commonsense reasoning in time (VisualCOMET)

Please find images in `t-manlingli/vcr/images/vcr1images` (using the same images as VCR), and the annotation data in `t-manlingli/visualcomet/visualcomet`.

## Code

The code structure for `CLIP-event` is as follows:
```
---- src
------------ CLIP-event  # our method
---------------- clip.py  # from OpenAI, the code for loading model and toeknizer
---------------- dataset_*  # data loaders
---------------- model_*  # models
---------------- engine.py  # training and testing engine
---------------- train.py  # training code
---------------- eval_*  # evaluation code
---------------- utils_*  # utils
```

## Model

Checkpoints are saved to [sdrgstd01scus](https://sdrgstd01scus.blob.core.windows.net/user/t-manlingli/checkpoint).

### Best performed models
The hyperparamters and training configuration is `config-ot-all-itp-a100-8gpu-template.json`, and the model checkpoints are in sdrgprmblob01scus:`t-manlingli/checkpoint/best`. 

## Preprocessing

### Text IE scripts
We follow [GAIA text pipeline](https://github.com/limanling/uiuc_ie_pipeline_fine_grained/blob/master/pipeline_full_en.sh) to extract entities and events from captions. The text IE scripts are detailed in `src/preprocess/voa/ie`:
```
---- ie
-------- set_up_m36.sh  # set up mongoDB for linking
-------- pipeline_full_en.sh  # English information extraction pipeline
-------- multimedia.sh  # vision information extraction pipeline
```

### Text OpenIE scripts
We use [Stanford OpenIE](https://nlp.stanford.edu/software/openie.html) to get the OpenIE results. The version we used is reverb.

### Vision IE scripts
We follow [GAIA multimedia pipeline](https://github.com/limanling/uiuc_ie_pipeline_fine_grained/blob/master/multimedia/multimedia.sh) to extract objects and visual events.


### IE visualizations
Please find the visualizations of text IE and openIE results in `t-manlingli/gsr/voa_caption_visualization`. The visualization code is `data/voa/visualization.py`.

### Negative description sample generation
The script is `src/preprocess/preprocess_description_contrastive.py`. We have two steps to generate the positive and negative descriptions:
- Salient event selection (`preprocess_event_selection` function): For each caption, we select one salient event for template choosing, based on event type frequency, the number of arguments, image-event similrity, etc. To enrich the event structure, we merge the arguments for events that have the same type. Also, we ignore image-caption pairs that do not have events. 
- Negative and positive sample generation (`neg_template` function): contains event-level negative samples and argument-level samples. 

## Training

There are two steps dring training:
- Prepare hypermeters and model loss function.

### Configurations
The configurations inclue:
```
{
    "task": "", # task name will be used as the name of the directories of outputs.
    "constrastive_loss": {"ce", "bce", "kl"},  # cross entropy, binary cross entropy, kl divergence
    "constrastive_overbatch": {true, false},
    "alignment": {true, false},
    "multiattention": {true, false},
    
    "posneg_descriptions_json": "", 
    "image_caption_json": [],   # a list of all image-captions.
    "image_dir": [],  # a list of all images
    

    "load_object": {true, false}, 
    "object_pickle": [],  
    "object_ontology_file": "src/clip/config/class-descriptions-boxable.csv", 
    "object_detection_threshold": {0-1}, 
    "object_topk": 50, 
    
    "load_ie": {true, false},
    "ie_ontology_json": "", 
    "input_entities": [],
    "input_events": [],
    "ltf_dir": "",
    
    "load_sr": {true, false}, 

    "sync_bn": {true, false},  # wherther synchronize batch normalization

    "ckpt_dir": "",  # checkpoint dir
    "tb_log_dir": "",  # tensorboard dir
    "print_freq": 1,
    "log_level": {"info", "debug"}, 

    "is_train": true,
    "begin_ckpt": "", 
    "jit": true,
    "begin_epoch": 0,
    "max_epoch": 30,
    "batch_size": 16,
    "lr": 1e-6,
    "optimizer": {"adam", "sgd"},
    "weight_decay": float, 
    "lr_scheduler": {"cosineannealinglr", "multisteplr", "warmup"}
    
}

```

### Training
Running on a sngle GPU:
```
python train.py --cfg ${config_file}
```
Running in distributed mode:
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py --cfg ${config_file}
```

## Testing

- Multimedia event extraction (M2E2)
```
python src/clip/CLIP-event/eval_m2e2.py
```

- Grounded situation recognition (GSR)
```
python src/clip/CLIP-event/eval_gsr.py
```

- Visual commonsense reasoning (VCR)
```
python src/clip/CLIP-event/eval_vcr.py
```

- Visual commonsense reasoning in time (VisualCOMET)
```
python src/clip/CLIP-event/eval_visualcomet.py
```
