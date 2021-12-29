python3 train_seg.py train \
--name coseg-vases \
--dataroot ./data/coseg-vases-MAPS-256-3 \
--upsample bilinear \
--batch_size 24 \
--parts 4 \
--augment_scale \
--arch deeplab \
--backbone resnet50 \
--lr 2e-2 
