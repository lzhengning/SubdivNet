python3.7 train_seg.py test \
--name HumanBody \
--dataroot ./data/HumanSeg-NS-256-3 \
--upsample bilinear \
--batch_size 24 \
--parts 8 \
--arch deeplab \
--backbone resnet50 \
--checkpoint ./checkpoints/HumanBody.pkl
