# CUDA_VISIBLE_DEVICES=0 
python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py --cfg "../config/config-ot-2009-devbox.json"