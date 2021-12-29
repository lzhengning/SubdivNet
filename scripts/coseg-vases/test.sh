python3 train_seg.py test \
--name coseg-vases \
--dataroot ./data/coseg-vases-MAPS-256-3 \
--upsample bilinear \
--batch_size 24 \
--parts 4 \
--arch deeplab \
--backbone resnet50 \
--checkpoint ./checkpoints/coseg-vases.pkl