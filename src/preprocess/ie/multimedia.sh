data_root='/shared/nas/data/m1/manling2/aida_docker_test/uiuc_ie_pipeline_fine_grained/data/sample_data/VOA_EN_NW_2017_sample50'
cu_toolbox='/shared/nas/data/m1/manling2/aida_docker_test/CU_toolbox_small'

docker run -it -v ${data_root}/vision:/root/input -v ${data_root}:/root/output -e CUDA_VISIBLE_DEVICES=0 --gpus=all yrf1/object-detection /bin/bash ./full_script.sh

docker run -it -v ${data_root}/vision:/root/LDC -v ${data_root}/ltf:/root/ltf -v ${data_root}:/root/shared -v ${cu_toolbox}:/root/models -e CUDA_VISIBLE_DEVICES=0 --gpus all limanling/grounding-merging /root/conda/envs/aida-env/bin/python Feature_Extraction.py
docker run -it -v ${data_root}/vision:/root/LDC -v ${data_root}/ltf:/root/ltf -v ${data_root}:/root/shared -v ${cu_toolbox}:/root/models -e CUDA_VISIBLE_DEVICES=0 --gpus all limanling/grounding-merging /root/conda/envs/aida-env/bin/python Visual_Grounding_mp.py
docker run -it -v ${data_root}/vision:/root/LDC -v ${data_root}/ltf:/root/ltf -v ${data_root}:/root/shared -v ${cu_toolbox}:/root/models -e CUDA_VISIBLE_DEVICES=0 --gpus all limanling/grounding-merging /root/conda/envs/aida-env/bin/python Graph_Merging.py