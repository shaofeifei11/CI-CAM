#!/usr/bin/env bash
# cub vgg16 train
python -u train2.py --gpu 3 --backbone vgg16 --danet 1 --backbone_rate 0.1 --function quadratic --decay_epoch 14 --decay_rate 0.5 --dataset cub --epoch 100 --batch_size 6 --non_local 1 --shared_classifier 1 --combination 1 --input_size 304 --seg_thr 0.0 --lr 0.0005 --decay 1 --attention 1 --pretrain 1 >> log/train_cub_vgg.txt 2>&1
# cub vgg16 test
python -u tmp_test.py --gpu 3 --backbone vgg16 --function quadratic --dataset cub --batch_size 6 --non_local 1 --shared_classifier 1 --combination 1  --input_size 304 --seg_thr 0.0 --attention 1 --pretrain 1 --model_path trained_weights/net_train_vgg16_cub.pth >> log/test_cub_vgg.txt 2>&1 &

# cub inceptionV3 train
python -u train2.py --gpu 3 --backbone inceptionV3 --crop_size 299 --attention_size 17 --backbone_rate 1.0 --function quadratic --decay_epoch 50 --decay_rate 0.5 --dataset cub --epoch 100 --batch_size 6 --non_local 1 --shared_classifier 1 --combination 1 --input_size 450 --seg_thr 0.0 --lr 0.0001 --decay 1 --attention 1 --pretrain 1 >> log/train_cub_inception_T_8_6.txt 2>&1 &
# cub inceptionV3 test
python -u tmp_test.py --gpu 2 --backbone inceptionV3 --crop_size 299 --attention_size 17 --backbone_rate 0.5 --function quadratic --decay_epoch 100 --decay_rate 0.5 --dataset cub --epoch 200 --batch_size 6 --non_local 1 --shared_classifier 1 --combination 1 --input_size 450 --seg_thr 0.0 --lr 0.0001 --decay 1 --attention 1 --pretrain 1 --model_path trained_weights/net_train_inceptionV3_cub.pth >> log/test_cub_inception.txt 2>&1 &



