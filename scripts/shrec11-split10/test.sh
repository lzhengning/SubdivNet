# test shrec11-split10
python3 train_cls.py test \
--name shrec11-split10 \
--dataroot ./data/SHREC11-MAPS-48-4-split10 \
--batch_size 64 \
--n_classes 30 \
--depth 4 \
--channels 32 64 64 128 128 \
--n_dropout 1 \
--checkpoint ./checkpoints/shrec11-split10.pkl
