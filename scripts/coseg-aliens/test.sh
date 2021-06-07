python3.7 train_seg.py test \
--name coseg-alien \
--dataroot ./data/coseg-aliens-MAPS-256-3 \
--upsample bilinear \
--batch_size 24 \
--parts 4 \
--arch deeplab \
--backbone resnet50 \
--checkpoint checkpoints/coseg-aliens.pkl