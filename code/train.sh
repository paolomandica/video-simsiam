####################################################################################################
# Data and Cache Paths
####################################################################################################

path_to_kinetics="/data_volume/data/kinetics/"
cache_path="/data_volume/data/cached_data/kinetics.pt"

path_to_kinetics_sample="/data_volume/data/kinetics_sample/"
cache_path_sample="/data_volume/data/cached_data/kinetics_sample.pt"


python -W ignore train.py --data-path $path_to_kinetics \
--cache-dataset --cache-path $cache_path \
--workers 20 --lr 0.05 --epochs 100 --batch-size 128 \
--clip-len 8 --frame-skip 32 --clips-step 256 --clips-per-video 1 \
--name "vfs" \
--data-parallel --visualize


# --output-dir "./checkpoints/simsiam/"
# --port 8094 
# --partial-reload "../pretrained.pth"
# --resume "./checkpoints/randomise_sp_unnorm/checkpoint.pth"