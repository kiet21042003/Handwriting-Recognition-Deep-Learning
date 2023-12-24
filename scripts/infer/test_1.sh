python test.py \
    --task validation \
    --root-dir '/mnt/disk1/nmduong/hust/intro2dl/data/new_train' \
    --train-label '/mnt/disk1/nmduong/hust/intro2dl/data/train_list.txt' \
    --test-label '/mnt/disk1/nmduong/hust/intro2dl/data/val_list.txt' \
    --img-width 100 \
    --img-height 32 \
    --lr 3e-4 \
    --decay-rate 0.9 \
    --num-epochs 500 \
    --lr-step-every 1000 \
    --max-length 25 \
    --batch-size 128 \
    --log-every 20 \
    --val-every 1000 \
    --weight './outputs/train_baseline_stn0_augment/best_cer.pth'
    # --wandb