import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import argparse
import random

import numpy as np
from tensorboardX import SummaryWriter

import jittor as jt
import jittor.nn as nn
from jittor.optim import Adam, SGD
from jittor.lr_scheduler import MultiStepLR
jt.flags.use_cuda = 1

from tqdm import tqdm

from subdivnet.dataset import SegmentationDataset
from subdivnet.deeplab import MeshDeepLab
from subdivnet.deeplab import MeshVanillaUnet
from subdivnet.utils import to_mesh_tensor
from subdivnet.utils import save_results
from subdivnet.utils import update_label_accuracy
from subdivnet.utils import compute_original_accuracy
from subdivnet.utils import SegmentationMajorityVoting


def train(net, optim, dataset, writer, epoch):
    net.train()
    acc = 0
    for meshes, labels, _ in tqdm(dataset, desc=str(epoch)):
        mesh_tensor = to_mesh_tensor(meshes)
        mesh_labels = jt.int32(labels)
        outputs = net(mesh_tensor)
        loss = nn.cross_entropy_loss(outputs.unsqueeze(dim=-1), mesh_labels.unsqueeze(dim=-1), ignore_index=-1)
        optim.step(loss)

        preds = np.argmax(outputs.data, axis=1)
        acc += np.sum((labels == preds).sum(axis=1) / meshes['Fs'])
        writer.add_scalar('loss', loss.data[0], global_step=train.step)
        train.step += 1
    acc /= dataset.total_len

    print('Epoch #{epoch}: train acc = ', acc)
    writer.add_scalar('train-acc', acc, global_step=epoch)


@jt.single_process_scope()
def test(net, dataset, writer, epoch, args):
    net.eval()
    acc = 0
    oacc = 0
    label_acc = np.zeros(args.parts)
    name = args.name
    voted = SegmentationMajorityVoting(args.parts, name)

    with jt.no_grad():
        for meshes, labels, mesh_infos in tqdm(dataset, desc=str(epoch)):
            mesh_tensor = to_mesh_tensor(meshes)
            mesh_labels = jt.int32(labels)
            outputs = net(mesh_tensor)
            preds = np.argmax(outputs.data, axis=1)

            batch_acc = (labels == preds).sum(axis=1) / meshes['Fs']
            batch_oacc = compute_original_accuracy(mesh_infos, preds, mesh_labels)
            acc += np.sum(batch_acc)
            oacc += np.sum(batch_oacc)
            update_label_accuracy(preds, mesh_labels, label_acc)
            voted.vote(mesh_infos, preds, mesh_labels)

    acc /= dataset.total_len
    oacc /= dataset.total_len
    voacc = voted.compute_accuracy(save_results=True)
    writer.add_scalar('test-acc', acc, global_step=epoch)
    writer.add_scalar('test-oacc', oacc, global_step=epoch)
    writer.add_scalar('test-voacc', voacc, global_step=epoch)

    # Update best results
    if test.best_oacc < oacc:
        if test.best_oacc > 0:
            os.remove(os.path.join('checkpoints', name, f'oacc-{test.best_oacc:.4f}.pkl'))
        net.save(os.path.join('checkpoints', name, f'oacc-{oacc:.4f}.pkl'))
        test.best_oacc = oacc

    if test.best_voacc < voacc:
        if test.best_voacc > 0:
            os.remove(os.path.join('checkpoints', name, f'voacc-{test.best_voacc:.4f}.pkl'))
        net.save(os.path.join('checkpoints', name, f'voacc-{voacc:.4f}.pkl'))
        test.best_voacc = voacc

    print('test acc = ', acc)
    print('test acc [original] =', oacc, ', best =', test.best_oacc)
    print('test acc [original] [voted] =', voacc, ', best =', test.best_voacc)
    print('test acc per label =', label_acc / dataset.total_len)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--optim', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[50, 100, 150])
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--upsample', choices=['nearest', 'bilinear'], default='bilinear')
    parser.add_argument('--parts', type=int, default=8)
    parser.add_argument('--augment_scale', action='store_true')
    parser.add_argument('--augment_orient', action='store_true')
    parser.add_argument('--arch', choices=['unet', 'deeplab', 'vunet'], default='unet')
    parser.add_argument('--backbone', choices=['resnet18', 'resnet50'], default='resnet50')
    parser.add_argument('--globalpool', choices=['max', 'mean'], default='mean')

    args = parser.parse_args()
    mode = args.mode
    name = args.name
    batch_size = args.batch_size
    dataroot = args.dataroot

    net = None
    if args.arch == 'deeplab':
        net = MeshDeepLab(13, args.parts, args.backbone, globalpool=args.globalpool)
    elif args.arch == 'unet':
        net = MeshVanillaUnet(13, args.parts, upsample=args.upsample)
    
    if args.optim == 'adam':
        optim = Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optim = SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optim, milestones=args.lr_milestones, gamma=args.lr_gamma)
 
    writer = SummaryWriter("logs/" + name)
    print('name:', name)

    augments = []
    if args.augment_scale:
        augments.append('scale')
    if args.augment_orient:
        augments.append('orient')

    if mode == 'train':
        train_dataset = SegmentationDataset(dataroot, batch_size=batch_size, 
            shuffle=True, train=True, num_workers=4, augments=augments)
    test_dataset = SegmentationDataset(dataroot, batch_size=8, shuffle=False, 
        train=False, num_workers=4)

    checkpoint_path = os.path.join('checkpoints', name)
    checkpoint_name = os.path.join(checkpoint_path, name + '-latest.pkl')
    os.makedirs(checkpoint_path, exist_ok=True)

    if args.checkpoint is not None:
        print('parameters: loaded from ', args.checkpoint)
        net.load(args.checkpoint)

    train.step = 0
    test.best_oacc = 0
    test.best_voacc = 0

    if args.mode == 'train':
        for epoch in range(500):
            train(net, optim, train_dataset, writer, epoch)
            test(net, test_dataset, writer, epoch, args)
            scheduler.step()
            net.save(checkpoint_name)
    else:
        test(net, test_dataset, writer, 0, args)
