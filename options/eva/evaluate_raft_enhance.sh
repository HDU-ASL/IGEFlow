CUDA_VISIBLE_DEVICES=0 python core/evaluate_one.py \
    --model runs/joint-raft-wave-vgg-load/checkpoints/1000_joint-raft-wave-vgg-load.pth \
    --dataset all \
    --result_path runs/eva/ced \
    --dataset_root /home/whx/code/Retinexformer/data/VBOF/VBOF_dataset \
    --enhance \
    # --vgg_loss
    