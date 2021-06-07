# train shrec11-split16
python3.7 train_cls.py train \
--name shrec11-split16 \
--dataroot ./data/SHREC11-MAPS-48-4-split16 \
--optim adam \
--lr 1e-3 \
--lr_milestones 50 100 \
--weight_decay 1e-4 \
--n_epoch 200 \
--batch_size 64 \
--n_classes 30 \
--depth 4 \
--channels 32 64 128 128 128 \
--n_dropout 1