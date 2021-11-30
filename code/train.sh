####################################################################################################
# Data and Cache Paths
####################################################################################################

path_to_kinetics="/data_volume/data/kinetics/"
cache_path="/data_volume/data/cached_data/kinetics.pt"

path_to_kinetics_sample="/data_volume/data/kinetics_sample/"
cache_path_sample="/data_volume/data/cached_data/kinetics_sample.pt"

####################################################################################################
# Core {Superpixels | Patches | Mix} Model Training
####################################################################################################

python -W ignore train.py --data-path $path_to_kinetics \
--cache-dataset --cache-path $cache_path \
--frame-aug grid --dropout 0.1 --clip-len 8 --temp 0.05 \
--model-type "scratch" --workers 40 --batch-size 40 --lr 0.05 \
--epochs 100 --data-parallel --visualize \
--output-dir "./checkpoints/simsiam/"
# --visualize --port 8094 
# --partial-reload "../pretrained.pth"
# --resume "./checkpoints/randomise_sp_unnorm/checkpoint.pth"
