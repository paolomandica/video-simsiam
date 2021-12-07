# Checkpoint to Evaluate
checkpoint="./checkpoints/simsiam_lstm/model_0.pth"

# ResNet layer to use
res_layer=4

# Results Output Directories
savepath="../results/masks/"
outpath="../results/converted/"

# Delete Contents of Results Directory
rm -rf $savepath
rm -rf $outpath

# Davis Paths
vallist="/data_volume/sapienza-video-contrastive/code/eval/davis_vallist.txt"
dataset="/data_volume/data/davis_val/"

# Run Evaluation
python test.py --filelist $vallist --model-type scratch \
--resume $checkpoint --save-path $savepath \
--topk 10 --videoLen 20 --radius 12 \
--temperature 0.05 --cropSize -1 --res-layer $res_layer

# Convert
python ./eval/convert_davis.py --in_folder $savepath \
--out_folder $outpath --dataset $dataset

# Compute metrics
python /data_volume/davis2017-evaluation/evaluation_method.py \
--task semi-supervised  --results_path $outpath \
--set val --davis_path $dataset
