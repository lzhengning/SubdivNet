# train shrec11-split10
python3.7 train_cls.py train \
--name shrec11-split10 \
--dataroot ./data/SHREC11-MAPS-48-4-split10 \
--optim adam \
--lr 1e-3 \
--lr_milestones 50 100 \
--weight_decay 1e-4 \
--n_epoch 200 \
--batch_size 64 \
--n_classes 30 \
--no_center_diff \
--depth 4 \
--channels 32 64 64 128 128 \
--n_dropout 1