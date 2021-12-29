# train cubes
python3 train_cls.py train \
--name cubes \
--dataroot ./data/Cubes-MAPS-48-4/ \
--optim adam \
--lr 1e-3 \
--lr_milestones 20 40 \
--n_epoch 60 \
--weight_decay 1e-4 \
--batch_size 64 \
--n_classes 22 \
--depth 4 \
--channels 32 64 128 128 128 \
--n_dropout 1
