# ***************************************************************
# Author:   Zheng-Ning Liu <lzhengning@gmail.com>
#
# The training & test script for mesh classification.
# ***************************************************************

import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from tensorboardX import SummaryWriter

import jittor as jt
import jittor.nn as nn
from jittor.optim import Adam
from jittor.optim import SGD
from jittor.lr_scheduler import MultiStepLR
jt.flags.use_cuda = 1

import numpy as np
from tqdm import tqdm

from subdivnet.dataset import ClassificationDataset
from subdivnet.network import MeshNet
from subdivnet.utils import to_mesh_tensor
from subdivnet.utils import ClassificationMajorityVoting


def train(net, optim, train_dataset, writer, epoch):
    net.train()
    n_correct = 0
    n_samples = 0

    jt.sync_all(True)

    disable_tqdm = jt.rank != 0
    for meshes, labels, _ in tqdm(train_dataset, desc=f'Train {epoch}', disable=disable_tqdm):

        mesh_tensor = to_mesh_tensor(meshes)
        mesh_labels = jt.int32(labels)

        outputs = net(mesh_tensor)
        loss = nn.cross_entropy_loss(outputs, mesh_labels)
        optim.step(loss)

        preds = np.argmax(outputs.data, axis=1)
        n_correct += np.sum(labels == preds)
        n_samples += outputs.shape[0]

        loss = loss.item()
        if jt.rank == 0:
            writer.add_scalar('loss', loss, global_step=train.step)

        train.step += 1

    # To avoid jittor handing when training with multiple gpus
    jt.sync_all(True)

    if jt.rank == 0:
        acc = n_correct / n_samples
        print('Epoch #{epoch}: train acc = ', acc)
        writer.add_scalar('train-acc', acc, global_step=epoch)


@jt.single_process_scope()
def test(net, test_dataset, writer, epoch, args):
    net.eval()
    acc = 0
    voted = ClassificationMajorityVoting(args.n_classes)
    with jt.no_grad():
        for meshes, labels, mesh_paths in tqdm(test_dataset, desc=f'Test {epoch}'):
            mesh_tensor = to_mesh_tensor(meshes)
            outputs = net(mesh_tensor)

            preds = np.argmax(outputs.data, axis=1)
            acc += np.sum(labels == preds)
            voted.vote(mesh_paths, preds, labels)

    acc /= test_dataset.total_len
    vacc = voted.compute_accuracy()

    # Update best results
    if test.best_acc < acc:
        if test.best_acc > 0:
            os.remove(os.path.join('checkpoints', name, f'acc-{test.best_acc:.4f}.pkl'))
        net.save(os.path.join('checkpoints', name, f'acc-{acc:.4f}.pkl'))
        test.best_acc = acc

    if test.best_vacc < vacc:
        if test.best_vacc > 0:
            os.remove(os.path.join('checkpoints', name, f'vacc-{test.best_vacc:.4f}.pkl'))
        net.save(os.path.join('checkpoints', name, f'vacc-{vacc:.4f}.pkl'))
        test.best_vacc = vacc

    print(f'Epoch #{epoch}: test acc = {acc}, best = {test.best_acc}')
    print(f'Epoch #{epoch}: test acc [voted] = {vacc}, best = {test.best_vacc}')
    writer.add_scalar('test-acc', acc, global_step=epoch)
    writer.add_scalar('test-vacc', vacc, global_step=epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--n_classes', type=int)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--optim', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=None)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--channels', type=int, nargs='+', required=True)
    parser.add_argument('--residual', action='store_true')
    parser.add_argument('--blocks', type=int, nargs='+', default=None)
    parser.add_argument('--n_dropout', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n_worker', type=int, default=4)
    parser.add_argument('--use_xyz', action='store_true')
    parser.add_argument('--use_normal', action='store_true')
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')

    args = parser.parse_args()
    mode = args.mode
    name = args.name
    dataroot = args.dataroot

    if args.seed is not None:
        jt.set_global_seed(args.seed)

    # ========== Dataset ==========
    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')
    train_dataset = ClassificationDataset(dataroot, batch_size=args.batch_size,
        shuffle=True, train=True, num_workers=args.n_worker, augment=augments)
    test_dataset = ClassificationDataset(dataroot, batch_size=args.batch_size,
        shuffle=False, train=False, num_workers=args.n_worker)

    input_channels = 7
    if args.use_xyz:
        train_dataset.feats.append('center')
        test_dataset.feats.append('center')
        input_channels += 3
    if args.use_normal:
        train_dataset.feats.append('normal')
        test_dataset.feats.append('normal')
        input_channels += 3

    # ========== Network ==========
    net = MeshNet(input_channels, out_channels=args.n_classes, depth=args.depth,
        layer_channels=args.channels, residual=args.residual,
        blocks=args.blocks, n_dropout=args.n_dropout)

    # ========== Optimizer ==========
    if args.optim == 'adam':
        optim = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = SGD(net.parameters(), lr=args.lr, momentum=0.9)

    if args.lr_milestones is not None:
        scheduler = MultiStepLR(optim, milestones=args.lr_milestones, gamma=0.1)
    else:
        scheduler = MultiStepLR(optim, milestones=[])

    # ========== MISC ==========
    if jt.rank == 0:
        writer = SummaryWriter("logs/" + name)
    else:
        writer = None

    checkpoint_path = os.path.join('checkpoints', name)
    checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.checkpoint is not None:
        print('parameters: loaded from ', args.checkpoint)
        net.load(args.checkpoint)

    train.step = 0
    test.best_acc = 0
    test.best_vacc = 0

    # ========== Start Training ==========
    if jt.rank == 0:
        print('name: ', name)

    if args.mode == 'train':
        for epoch in range(args.n_epoch):
            train(net, optim, train_dataset, writer, epoch)
            test(net, test_dataset, writer, epoch, args)
            scheduler.step()

            jt.sync_all()
            if jt.rank == 0:
                net.save(checkpoint_name)
    else:
        test(net, test_dataset, writer, 0, args)
