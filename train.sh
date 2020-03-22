#/bin/bash
DATA_ROOT=/data/datasets/im/pytorchdim/Combined_Dataset
TRAIN_DATA_ROOT=$DATA_ROOT/Training_set/comp
TEST_DATA_ROOT=$DATA_ROOT/Test_set/comp

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
    --testImgDir=$TEST_DATA_ROOT/image \
    --testTrimapDir=$TEST_DATA_ROOT/trimap \
    --testAlphaDir=$TEST_DATA_ROOT/alpha \
    --testResDir=result/tmp \
    --crop_or_resize=whole \
    --max_size=1600 \
    --write_log \
    --debug \

    #--resume=model/stage1/ckpt_e1.pth \
