#/bin/bash

python main.py \
    --mode=train \
    --crop_h=320,480,640 \
    --crop_w=320,480,640 \
    --size_h=320 \
    --size_w=320 \
    --batch_size=2 \
    --epochs=120 \
    --lr=0.0001 \
    --cuda \
    --gpu=4 \
    --max_size=1600 \
    --write_log \
    --save_ckpt

    #--resume=model/stage1/ckpt_e1.pth \
