CUDA_VISIBLE_DEVICES=1 python core/evaluate_one.py \
    --model runs/debug/raft/raft-chairs.pth \
    --dataset all \
    --result_path runs/eva/raft \
    --dataset_root /home/whx/code/Retinexformer/data/VBOF/VBOF_dataset \
    