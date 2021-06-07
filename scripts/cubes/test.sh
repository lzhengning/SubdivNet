# train cubes
python3.7 train_cls.py test \
--name Cubes \
--dataroot ./data/Cubes-MAPS-48-4/ \
--batch_size 64 \
--n_classes 22 \
--depth 4 \
--channels 32 64 128 128 128 \
--n_dropout 1 \
--checkpoint ./checkpoints/Cubes.pkl
