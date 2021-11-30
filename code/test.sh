# Checkpoint to Evaluate
checkpoint="./checkpoints/_drop0.1-len4-ftranscrop-faugnone-optimadam-temp0.05-fdrop0.0-lr0.0003-mlp0-spslic-nsp16-p0.0/model_0.pth"

# Results Output Directories
savepath="../results/masks/"
outpath="../results/converted/"

# Davis Paths
vallist="/data_volume/sapienza-video-contrastive/code/eval/davis_vallist.txt"
dataset="/data_volume/data/davis_val/"

# Delete Contents of Results Directory
rm -rf ../results/*

# # Pretrained Model Paths
# pretrained_checkpoint="../pretrained.pth"
# pretrained_savepath="../results/pretrained/"
# pretrained_outpath="../results/pretrained_converted/"

# Run Evaluation
python test.py --filelist $vallist --model-type scratch \
--resume $checkpoint --save-path $savepath \
--topk 10 --videoLen 20 --radius 12  --temperature 0.05  --cropSize -1 

# Convert
python ./eval/convert_davis.py --in_folder $savepath \
--out_folder $outpath --dataset $dataset

# Compute metrics
python /data_volume/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path $outpath \
--set val --davis_path $dataset
