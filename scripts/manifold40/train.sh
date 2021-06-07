# Train
python3.7 train_cls.py train \
--name manifold40 \
--dataroot ./data/Manifold40-MAPS-96-3/ \
--optim adam \
--lr 1e-3 \
--lr_milestones 20 40 \
--batch_size 48 \
--n_classes 40 \
--depth 3 \
--channels 32 64 128 256 \
--n_dropout 2 \
--use_xyz \
--use_normal \
--augment_scale
