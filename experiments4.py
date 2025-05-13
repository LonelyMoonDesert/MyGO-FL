import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import argparse
import logging
import os
import copy
from math import *
import random

import datetime
# from torch.utils.tensorboard import SummaryWriter

from model import *
from utils import *
from vggmodel import *
from resnetcifar import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', help='neural network used in training')
    parser.add_argument('--resume', type=bool, default=True, help='whether to resume training')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='directory to save checkpoints')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset used for training')
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))))
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--lr_G', type=float, default=0.01, help='learning rate (default: 0.001)')
    parser.add_argument('--lr_D', type=float, default=0.01, help='learning rate (default: 0.005)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--epoch_G', type=int, default=5, help='number of local epochs')
    parser.add_argument('--epoch_D', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=2,  help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                            help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--training_type', type=str, default='local', help='local/adversarial')
    parser.add_argument('--use_projection_head', type=bool, default=False, help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1, help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--l1_lambda', type=float, default=1e-4, help="L1 regularization strength")
    parser.add_argument('--l1', type=bool, default=False, help="Use L1 regularization")
    parser.add_argument('--adv_l1_lambda', type=float, default=1e-4, help="L1 regularization strength")
    parser.add_argument('--adv_l1', type=bool, default=False, help="Use L1 regularization")
    parser.add_argument('--lambda_adv', type=float, default=0.3, help="adv_loss")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5, help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--optimizer_G', type=str, default='amsgrad', help='the optimizer')
    parser.add_argument('--optimizer_D', type=str, default='amsgrad', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level', help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')

    # for scheduler
    parser.add_argument('--scheduler', type=str, default='StepLR', help='Type of scheduler: StepLR, MultiStepLR, etc.')
    parser.add_argument('--scheduler_g', type=str, default='StepLR', help='Type of scheduler: StepLR, MultiStepLR, etc.')
    parser.add_argument('--scheduler_d', type=str, default='StepLR', help='Type of scheduler: StepLR, MultiStepLR, etc.')
    parser.add_argument('--gamma', type=float, default=0.01, help='Decay rate for learning rate.')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for StepLR.')
    parser.add_argument('--milestones', type=str, default='30,60',
                        help='Milestones for MultiStepLR, separated by commas.')
    parser.add_argument('--T_max', type=int, default=50, help='Maximum number of iterations for CosineAnnealingLR.')
    parser.add_argument('--factor', type=float, default=0.1,
                        help='Factor by which the learning rate will be reduced. ReduceLROnPlateau.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs with no improvement after which learning rate will be reduced. ReduceLROnPlateau.')
    parser.add_argument('--base_lr', type=float, default=0.001, help='Initial learning rate for CyclicLR.')
    parser.add_argument('--max_lr', type=float, default=0.01, help='Maximum learning rate for CyclicLR.')
    parser.add_argument('--step_size_up', type=int, default=2000,
                        help='Number of training iterations in the increasing half of a cycle. CyclicLR.')
    parser.add_argument('--steps_per_epoch', type=int, default=1000,
                        help='Number of steps per epoch, used for OneCycleLR.')

    args = parser.parse_args()
    return args


def init_nets(net_configs, dropout_p, n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}

    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist'}:
        n_classes = 10
    elif args.dataset == 'celeba':
        n_classes = 2
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    elif args.dataset == 'femnist':
        n_classes = 62
    elif args.dataset == 'emnist':
        n_classes = 47
    elif args.dataset in {'a9a', 'covtype', 'rcv1', 'SUSY'}:
        n_classes = 2

    if args.use_projection_head:
        add = ""
        if "mnist" in args.dataset and args.model == "simple-cnn":
            add = "-mnist"
        for net_i in range(n_parties):
            net = ModelFedCon(args.model + add, args.out_dim, n_classes, net_configs)
            nets[net_i] = net
    else:
        if args.alg == 'moon':
            add = ""
            if "mnist" in args.dataset and args.model == "simple-cnn":
                add = "-mnist"
            for net_i in range(n_parties):
                net = ModelFedCon_noheader(args.model + add, args.out_dim, n_classes, net_configs)
                nets[net_i] = net
        else:
            for net_i in range(n_parties):
                if args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16, 8]
                    net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                elif args.model == "vgg11":
                    net = vgg11()
                    hook_handle = net.features[20].register_forward_hook(hook_fn)
                elif args.model == "simple-cnn":
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
                        if args.dataset == "cifar10":
                            hook_handle = net.conv2.register_forward_hook(hook_fn)
                    elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                        net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
                        hook_handle = net.conv2.register_forward_hook(hook_fn)
                    elif args.dataset == 'celeba':
                        net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)
                        hook_handle = net.conv2.register_forward_hook(hook_fn)
                elif args.model == "vgg9":
                    if args.dataset in ("mnist", 'femnist'):
                        net = ModerateCNNMNIST()
                        hook_handle = net.conv_layer[4].register_forward_hook(hook_fn)
                    elif args.dataset in ("cifar10", "cinic10", "svhn"):
                        # print("in moderate cnn")
                        net = ModerateCNN()
                        hook_handle = net.conv_layer[4].register_forward_hook(hook_fn)
                    elif args.dataset == 'celeba':
                        net = ModerateCNN(output_dim=2)
                        hook_handle = net.conv_layer[4].register_forward_hook(hook_fn)
                elif args.model == "resnet50":
                    net = ResNet50_cifar10()
                    hook_handle = net.layer3.register_forward_hook(hook_fn_resnet50_cifar10)
                elif args.model == "resnet18":
                    net = ResNet18_cifar10()
                    hook_handle = net.layer3.register_forward_hook(hook_fn_reduce_feature_map)
                elif args.model == "vgg16":
                    net = vgg16()
                else:
                    print("not supported yet")
                    exit(1)
                nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)
    return nets, model_meta_data, layer_type


def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    G_output_list = None
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_l1_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)

                # 在最后一个 epoch 记录下 G_output
                if epoch == epochs - 1:
                    # feature_map = feature_map.cpu()  # Ensure feature_map is on CPU
                    if G_output_list is None:
                        G_output_list = feature_map  # Initialize with the first feature map
                    else:
                        # Concatenate along the batch dimension
                        G_output_list = torch.cat((G_output_list, feature_map), dim=0)

                l1_norm = sum(p.abs().sum() for p in net.parameters())
                if args.l1:
                    loss = criterion(out, target) + 0.01 * args.l1_lambda * l1_norm
                    # print("with l1 loss")
                else:
                    loss = criterion(out, target)
                # loss = criterion(out, target)
                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
                if args.l1:
                    epoch_l1_loss_collector.append(0.01 * args.l1_lambda * l1_norm)

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        if args.l1:
            epoch_l1_loss = sum(epoch_l1_loss_collector) / len(epoch_l1_loss_collector)
            logger.info('Epoch: %d Loss: %f L1 loss: %f' % (epoch, epoch_loss, epoch_l1_loss))
        else:
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    logger.info('Shape of G_output_list: {}'.format(G_output_list.shape if G_output_list is not None else 0))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')

    # Save the training state
    checkpoint = {
        'client': net_id,
        'model': net.state_dict(),  # Corrected key to 'model_state'
        'round': round
    }
    filename = f"client{net_id}_round{round}.pth"
    save_checkpoint(checkpoint, './checkpoints', filename)

    return train_acc, test_acc, G_output_list


# 定义对抗训练的函数，用于在客户端上更新生成器
# nets: 客户端模型列表
# d_model: 判别器模型
# lambda_adv: 调节系数，用于平衡任务损失和对抗损失
# args: 其他参数（例如学习率等）
# net_dataidx_map: 数据索引映射
# train_dl: 本地训练数据加载器
# device: 设备（例如 "cpu" 或 "cuda"）
def adv_train_net(net_id, net, D, lambda_adv, train_dataloader, test_dataloader, epochs, lr, args_optimizer,
                  device="cpu"):
    logger.info('Starting adversarial training of clients...')
    criterion_task = nn.CrossEntropyLoss().to(device)
    criterion_adv = nn.BCELoss().to(device)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # 初始化优化器，只对生成器部分的参数进行优化
    # 获取生成器部分的参数（conv1 到 layer3 之前的所有层）
    generator_parameters = []
    G_output_list = None
    for name, param in net.named_parameters():
        if 'layer4' in name or 'avgpool' in name or 'fc' in name:
            param.requires_grad = False
        else:
            generator_parameters.append(param)

    optimizer = optim.SGD(generator_parameters, lr=lr, momentum=args.rho, weight_decay=args.reg)

    if args_optimizer == 'adam':
        optimizer = optim.Adam(generator_parameters, lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(generator_parameters, lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(generator_parameters, lr=lr, momentum=args.rho, weight_decay=args.reg)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # 训练循环
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_task_loss_collector = []
        epoch_adv_loss_collector = []
        epoch_l1_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x = x.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long().view(-1)

                # 生成伪标签
                # fake_labels = torch.zeros((x.shape[0]), 1).to(device).view(-1)
                fake_labels = torch.full((x.shape[0],), 0.1, device=device)
                real_labels = torch.full((x.shape[0],), 0.9, device=device)

                # 前向传播，计算生成器输出
                out = net(x)  # 通过模型的前半部分

                # 在最后一个 epoch 记录下 G_output
                if epoch == epochs - 1:
                    # feature_map = feature_map.cpu()  # Ensure feature_map is on CPU
                    if G_output_list is None:
                        G_output_list = feature_map  # Initialize with the first feature map
                    else:
                        # Concatenate along the batch dimension
                        G_output_list = torch.cat((G_output_list, feature_map), dim=0)

                # 获取生成器的中间特征图 (通过 hook 获得 feature_map)
                D_out = D(feature_map).squeeze()
                # print('D_out: {}'.format(D_out))
                # print('fake_labels: {}'.format(fake_labels))

                # 计算任务损失和对抗损失
                task_loss = criterion_task(out, target)
                # adv_loss = criterion_adv(D_out, fake_labels)
                adv_loss = criterion_adv(D_out, real_labels)

                # 计算总损失
                l1_norm = sum(p.abs().sum() for p in net.parameters())

                # 计算总损失
                if args.l1:
                    loss = (1 - lambda_adv) * task_loss + lambda_adv * adv_loss + 0.01 * args.l1_lambda * l1_norm
                else:
                    loss = (1 - lambda_adv) * task_loss + lambda_adv * adv_loss
                # loss = 10 * adv_loss
                loss.backward()

                # 更新生成器参数
                optimizer.step()

                epoch_loss_collector.append(loss.item())
                epoch_task_loss_collector.append(task_loss.item())
                epoch_adv_loss_collector.append(adv_loss.item())
                if args.l1:
                    epoch_l1_loss_collector.append(0.01 * args.l1_lambda * l1_norm)

        # 记录每个epoch最后一个batch的损失
        # logger.info('Epoch: %d Last Batch Total Loss: %f Task Loss: %f Adversarial Loss: %f' % (
        # epoch, epoch_loss_collector[-1], epoch_task_loss_collector[-1], epoch_adv_loss_collector[-1]))

        # 记录每个epoch最后一个batch的损失
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_task_loss = sum(epoch_task_loss_collector) / len(epoch_task_loss_collector)
        epoch_adv_loss = sum(epoch_adv_loss_collector) / len(epoch_adv_loss_collector)


        if args.l1:
            epoch_l1_loss = sum(epoch_l1_loss_collector) / len(epoch_l1_loss_collector)
            logger.info('Epoch: %d Last Batch Total Loss: %f Task Loss: %f Adversarial Loss: %f L1 norm: %f' % (epoch, epoch_loss, epoch_task_loss, epoch_adv_loss, epoch_l1_loss))
        else:
            logger.info('Epoch: %d Last Batch Total Loss: %f Task Loss: %f Adversarial Loss: %f ' % (epoch, epoch_loss, epoch_task_loss, epoch_adv_loss))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('Shape of G_output_list: {}'.format(G_output_list.shape if G_output_list is not None else 0))

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')

    # 保存模型
    # Save the training state
    checkpoint = {
        'client': net_id,
        'model': net.state_dict(),  # Corrected key to 'model_state'
        'round': round
    }
    filename = f"adv_client{net_id}_round{round}.pth"
    save_checkpoint(checkpoint, './checkpoints', filename)
    return train_acc, test_acc, G_output_list


def train_net_fedgan(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    G_output_list = None
    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_l1_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)

                # 在最后一个 epoch 记录下 G_output
                if epoch == epochs - 1:
                    # feature_map = feature_map.cpu()  # Ensure feature_map is on CPU
                    if G_output_list is None:
                        G_output_list = feature_map  # Initialize with the first feature map
                    else:
                        # Concatenate along the batch dimension
                        G_output_list = torch.cat((G_output_list, feature_map), dim=0)

                l1_norm = sum(p.abs().sum() for p in net.parameters())
                if args.l1:
                    loss = criterion(out, target) + 0.01 * args.l1_lambda * l1_norm
                    # print("with l1 loss")
                else:
                    loss = criterion(out, target)

                # loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())
                if args.l1:
                    epoch_l1_loss_collector.append(0.01 * args.l1_lambda * l1_norm)

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        if args.l1:
            epoch_l1_loss = sum(epoch_l1_loss_collector) / len(epoch_l1_loss_collector)
            logger.info('Epoch: %d Loss: %f L1 loss: %f' % (epoch, epoch_loss, epoch_l1_loss))
        else:
            logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    logger.info('Shape of G_output_list: {}'.format(G_output_list.shape if G_output_list is not None else 0))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')

    # Save the training state
    checkpoint = {
        'client': net_id,
        'model': net.state_dict(),  # Corrected key to 'model_state'
        'round': round
    }
    filename = f"client{net_id}_round{round}.pth"
    save_checkpoint(checkpoint, './checkpoints', filename)

    return train_acc, test_acc, G_output_list


def train_net_fedavg(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # train_acc = compute_accuracy(net, train_dataloader, device=device)
        # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

        # writer.add_scalar('Accuracy/train', train_acc, epoch)
        # writer.add_scalar('Accuracy/test', test_acc, epoch)

        # if epoch % 10 == 0:
        #     logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_fedprox(net_id, net, global_net, train_dataloader, test_dataloader, epochs, lr, args_optimizer, mu,
                      device="cpu"):
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    # mu = 0.001
    global_weight_collector = list(global_net.to(device).parameters())

    for epoch in range(epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            # for fedprox
            fed_prox_reg = 0.0
            for param_index, param in enumerate(net.parameters()):
                fed_prox_reg += ((mu / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
            loss += fed_prox_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        # if epoch % 10 == 0:
        #     train_acc = compute_accuracy(net, train_dataloader, device=device)
        #     test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
        #
        #     logger.info('>> Training accuracy: %f' % train_acc)
        #     logger.info('>> Test accuracy: %f' % test_acc)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr,
                       args_optimizer, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    c_local.to(device)
    c_global.to(device)
    global_model.to(device)

    c_global_para = c_global.state_dict()
    c_local_para = c_local.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])
                net.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    c_new_para = c_local.state_dict()
    c_delta_para = copy.deepcopy(c_local.state_dict())
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (
                    cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.load_state_dict(c_new_para)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, c_delta_para


def train_net_fednova(net_id, net, global_model, train_dataloader, test_dataloader, epochs, lr, args_optimizer,
                      device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                          weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    # writer = SummaryWriter()

    tau = 0

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                tau = tau + 1

                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    global_model.to(device)
    a_i = (tau - args.rho * (1 - pow(args.rho, tau)) / (1 - args.rho)) / (1 - args.rho)
    global_model.to(device)
    global_model_para = global_model.state_dict()
    net_para = net.state_dict()
    norm_grad = copy.deepcopy(global_model.state_dict())
    for key in norm_grad:
        # norm_grad[key] = (global_model_para[key] - net_para[key]) / a_i
        norm_grad[key] = torch.true_divide(global_model_para[key] - net_para[key], a_i)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc, a_i, norm_grad


def train_net_moon(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                   args_optimizer, mu, temperature, args,
                   round, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True,
                                             device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # conloss = ContrastiveLoss(temperature)

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().to(device)
    # global_net.to(device)

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to(device)
    global_w = global_net.state_dict()
    # oppsi_nets = copy.deepcopy(previous_nets)
    # for net_id, oppsi_net in enumerate(oppsi_nets):
    #     oppsi_w = oppsi_net.state_dict()
    #     prev_w = previous_nets[net_id].state_dict()
    #     for key in oppsi_w:
    #         oppsi_w[key] = 2*global_w[key] - prev_w[key]
    #     oppsi_nets.load_state_dict(oppsi_w)
    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1).to(device)
    # mu = 0.001

    for epoch in range(epochs):
        epoch_loss_collector = []
        epoch_loss1_collector = []
        epoch_loss2_collector = []
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            if target.shape[0] == 1:
                continue

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            _, pro1, out = net(x)
            _, pro2, _ = global_net(x)
            if args.loss == 'l2norm':
                loss2 = mu * torch.mean(torch.norm(pro2 - pro1, dim=1))

            elif args.loss == 'only_contrastive' or args.loss == 'contrastive':
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                for previous_net in previous_nets:
                    previous_net.to(device)
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                    # previous_net.to('cpu')

                logits /= temperature
                labels = torch.zeros(x.size(0)).to(device).long()

                # loss = criterion(out, target) + mu * ContraLoss(pro1, pro2, pro3)

                loss2 = mu * criterion(logits, labels)

            if args.loss == 'only_contrastive':
                loss = loss2
            else:
                loss1 = criterion(out, target)
                loss = loss1 + loss2

            loss.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))

    if args.loss != 'l2norm':
        for previous_net in previous_nets:
            previous_net.to('cpu')
    train_acc = compute_accuracy(net, train_dataloader, moon_model=True, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, moon_model=True,
                                             device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')
    return train_acc, test_acc


def view_image(train_dataloader):
    for (x, target) in train_dataloader:
        np.save("img.npy", x)
        print(x.shape)
        exit(0)


feature_map = None


# 定义 hook 函数
def hook_fn(module, input, output):
    global feature_map
    feature_map = output  # 将中间输出存储到全局变量中


def hook_fn_to_2d(module, input, output):
    global feature_map
    # 全局平均池化
    pooled_output = nn.functional.adaptive_avg_pool2d(output, (1, 1))
    # 展平为二维，形状为 (batch_size, channels)
    flattened_output = pooled_output.view(output.size(0), -1)
    # 使用线性层降维，将通道数从 256 降到 2
    linear = nn.Linear(256, 2).to(output.device)
    feature_map = linear(flattened_output)


# 定义 hook 函数，进行特征图降维
def hook_fn_reduce_feature_map(module, input, output):
    global feature_map
    # 使用卷积降维，将通道数从 256 降到 m，空间维度从 (8, 8) 到 (n, n)
    m, n = 64, 4  # 可以根据需要调整 m 和 n 的值
    # 卷积层用于减少通道数
    conv = nn.Conv2d(in_channels=256, out_channels=m, kernel_size=3, stride=2, padding=1).to(output.device)
    reduced_output = conv(output)
    # 使用自适应平均池化调整空间维度为 (n, n)
    reduced_output = nn.functional.adaptive_avg_pool2d(reduced_output, (n, n))
    feature_map = reduced_output


# 定义 hook 函数，resnet50-cifar10
def hook_fn_resnet50_cifar10(module, input, output):
    global feature_map
    # 使用卷积降维，将通道数从 1024 降到 m，空间维度从 (8, 8) 到 (n, n)
    m, n = 64, 4  # 可以根据需要调整 m 和 n 的值
    # 卷积层用于减少通道数，这里将 in_channels 设置为 1024，匹配输入的通道数
    conv = nn.Conv2d(in_channels=1024, out_channels=m, kernel_size=3, stride=2, padding=1).to(output.device)

    # 先用卷积降维通道数
    reduced_output = conv(output)

    # 使用自适应平均池化调整空间维度为 (n, n)
    reduced_output = nn.functional.adaptive_avg_pool2d(reduced_output, (n, n))

    # 保存特征图
    feature_map = reduced_output


def save_global_model(global_model_checkpoint, directory, filename="global_model.pth"):
    """
    Save the global model state to a specified directory with a consistent filename format.
    This also saves the round number for tracking the training progress.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    torch.save(global_model_checkpoint, filepath)
    print("Global model saved to {}".format(filepath))


def save_checkpoint(state, checkpoint_dir, filename="checkpoint.pth.tar"):
    """
    Saves the training checkpoint to a specified directory with a given filename.
    Ensures that the directory structure is consistent across various saving functions.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists.
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print("Checkpoint saved to {}".format(filepath))

def split_G_output_by_clients(G_output_list_all_clients, net_dataidx_map):
    client_tensors = {}
    start_idx = 0

    # 遍历每个客户端的索引，依据样本数量来进行划分
    for net_id, dataidxs in net_dataidx_map.items():
        n_samples = len(dataidxs)
        end_idx = start_idx + n_samples

        # 按照样本数量划分 G_output_list_all_clients
        client_tensors[net_id] = G_output_list_all_clients[start_idx:end_idx]

        # 更新下一个客户端的起始索引
        start_idx = end_idx

    return client_tensors


# 新增的判别器训练函数
def train_discriminator(D, G_output_list_all_clients, real_data, net_dataidx_map, device, args_optimizer, lr, epochs=5,
                        batch_size=64):
    D.train()  # 确保判别器处于训练模式
    criterion = nn.BCELoss()  # 二元交叉熵损失

    # 根据优化器类型选择优化器
    if args_optimizer == 'adam':
        optimizer = optim.Adam(D.parameters(), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(D.parameters(), lr=lr, weight_decay=args.reg, amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(D.parameters(), lr=lr, momentum=args.rho, weight_decay=args.reg)
    else:
        optimizer = optim.SGD(D.parameters(), lr=lr, momentum=args.rho, weight_decay=args.reg)

    # 准备假数据和真实数据
    # client_tensors = split_G_output_by_clients(G_output_list_all_clients, net_dataidx_map)
    # fake_data = torch.cat([tensor for _, tensor in client_tensors.items()], dim=0).to(device)  # 将假数据迁移到指定设备
    fake_data = G_output_list_all_clients.to(device)
    real_data = real_data.to(device)  # 将真实数据迁移到指定设备

    # 创建数据加载器，应用标签平滑
    fake_dataset = torch.utils.data.TensorDataset(fake_data,
                                                  torch.full((len(fake_data),), 0.1).to(device))  # 假数据标签为0.1，并迁移到设备
    real_dataset = torch.utils.data.TensorDataset(real_data,
                                                  torch.full((len(real_data),), 0.9).to(device))  # 真实数据标签为0.9，并迁移到设备

    # 合并数据集
    combined_dataset = torch.utils.data.ConcatDataset([fake_dataset, real_dataset])
    data_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=False)
    fake_data_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=batch_size, shuffle=True)
    real_data_loader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs_batch, targets_batch) in enumerate(fake_data_loader):
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
            inputs_batch = inputs_batch.clone().detach().requires_grad_(True)  # 创建一个叶子节点，并设置requires_grad为True
            # 确保targets_batch的形状与outputs匹配，调整为 [batch_size]
            targets_batch = targets_batch.view(-1)  # 将targets_batch调整为一维张量

            optimizer.zero_grad()

            # print(targets_batch.shape)  # torch.Size([64])
            # print(inputs_batch.shape) # torch.Size([64, 64, 14, 14])
            outputs = D(inputs_batch)
            # print(outputs.shape)    # torch.Size([1024])
            outputs = outputs.squeeze()  # 确保outputs为 [batch_size] 形状

            fake_loss = criterion(outputs, targets_batch)

            # print('outputs: ', outputs)
            # print('targets: ', targets_batch)

            fake_loss.backward()
            optimizer.step()

            epoch_loss += fake_loss.item()

            # 计算准确率
            predicted = (outputs >= 0.5).float()
            # print('predicted: ', predicted)
            total += targets_batch.size(0)
            correct += torch.isclose(predicted, targets_batch, atol=0.1).sum().item()

        for batch_idx, (inputs_batch, targets_batch) in enumerate(real_data_loader):
            inputs_batch, targets_batch = inputs_batch.to(device), targets_batch.to(device)
            inputs_batch = inputs_batch.clone().detach().requires_grad_(True)  # 创建一个叶子节点，并设置requires_grad为True
            # 确保targets_batch的形状与outputs匹配，调整为 [batch_size]
            targets_batch = targets_batch.view(-1)  # 将targets_batch调整为一维张量

            optimizer.zero_grad()

            outputs = D(inputs_batch)
            outputs = outputs.squeeze()  # 确保outputs为 [batch_size] 形状
            real_loss = criterion(outputs, targets_batch)

            # print('outputs: ', outputs)
            # print('targets: ', targets_batch)

            real_loss.backward()
            optimizer.step()

            epoch_loss += real_loss.item()

            # 计算准确率
            predicted = (outputs >= 0.5).float()
            # print('predicted: ', predicted)
            total += targets_batch.size(0)
            correct += torch.isclose(predicted, targets_batch, atol=0.1).sum().item()

        avg_loss = epoch_loss / len(data_loader)
        accuracy = 100 * correct / total
        logger.info(f'Epoch {epoch + 1}/{epochs}, Discriminator Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    logger.info(' ** Discriminator training complete **')
    D.eval()  # 训练结束后将判别器设置为评估模式


def get_feature_maps(nets, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    global feature_map  # 声明为全局变量
    G_output_list_all_clients = None

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Getting feature maps of network %s. n_feature_maps: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)

        G_output_list = []

        if not isinstance(train_dl_local, list):
            train_dl_local = [train_dl_local]

        for tmp in train_dl_local:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                out = net(x)

                # 假设 feature_map 是 net 的某一层的输出（你可能需要使用 hook 或其他方式来获取）
                # 将每个批次的 feature_map 转移到 CPU 以节省 GPU 内存
                # feature_map = out.detach().cpu()
                mid_output = feature_map
                feature_map = mid_output.detach().cpu()

                # 将 feature_map 添加到列表中，而不是在 GPU 上拼接
                G_output_list.append(feature_map)

        # 将整个客户端的特征图拼接为一个大的张量
        G_output_list = torch.cat(G_output_list, dim=0)

        logger.info('Shape of G_output_list: {}'.format(G_output_list.shape if G_output_list is not None else 0))

        net.to('cpu')
        logger.info(' ** Get feature maps complete **')

        if G_output_list_all_clients is None:
            G_output_list_all_clients = G_output_list
        else:
            G_output_list_all_clients = torch.cat((G_output_list_all_clients, G_output_list), dim=0)

        logger.info('>> Shape of G_output_list_all_clients: {}'.format(G_output_list_all_clients.shape))

    return G_output_list_all_clients


def update_client_task_layers(global_model, client_models):
    # 提取全局模型的状态字典
    global_state_dict = global_model.state_dict()

    # 定义任务相关层的前缀
    task_layers_prefixes = ['layer4', 'fc']  # 假设'fc'是最后的全连接层

    # 更新每个客户端模型
    for client_model in client_models.values():
        client_state_dict = client_model.state_dict()

        # 遍历全局模型的参数，更新任务相关层
        for key, value in global_state_dict.items():
            if any(key.startswith(prefix) for prefix in task_layers_prefixes):
                client_state_dict[key] = value

        # 加载更新后的状态字典到客户端模型
        client_model.load_state_dict(client_state_dict, strict=False)


def aggregate_task_layers_weighted(global_model, client_models, net_dataidx_map, selected_clients):
    """
    按照客户端数据比例加权聚合客户端的任务部分参数，并将结果更新到全局模型中。

    参数：
    - global_model: 全局模型
    - client_models: 客户端模型字典，键为客户端标识，值为客户端模型
    - net_dataidx_map: 一个字典，键为客户端标识，值为每个客户端的数据索引
    - selected_clients: 被选中的客户端列表

    返回：
    - 无，直接更新 global_model
    """
    # 提取全局模型的状态字典
    global_state_dict = global_model.state_dict()

    # 定义任务相关层的前缀
    task_layers_prefixes = ['layer4', 'fc']  # 假设 'fc' 是最后的全连接层

    # 初始化聚合参数字典
    aggregated_params = {key: torch.zeros_like(value) for key, value in global_state_dict.items() if
                         any(key.startswith(prefix) for prefix in task_layers_prefixes)}

    # 计算每个客户端的数据比例（权重）
    total_data_points = sum([len(net_dataidx_map[client]) for client in selected_clients])
    fed_avg_freqs = [len(net_dataidx_map[client]) / total_data_points for client in selected_clients]

    # 遍历每个选中的客户端，进行加权聚合
    for idx, client_id in enumerate(selected_clients):
        client_model = client_models[client_id]
        client_state_dict = client_model.state_dict()

        for key in aggregated_params.keys():
            if idx == 0:
                # 初始化聚合参数
                aggregated_params[key] = client_state_dict[key] * fed_avg_freqs[idx]
            else:
                # 加权累加
                aggregated_params[key] += client_state_dict[key] * fed_avg_freqs[idx]

    # 更新全局模型的任务层参数
    global_state_dict.update(aggregated_params)
    global_model.load_state_dict(global_state_dict, strict=False)

    return global_model


# 定义生成真样本的函数
def generate_real_samples(global_model, data_loader, device="cpu"):
    global_model.eval()  # 设置为评估模式，不会更新参数
    real_samples = []

    with torch.no_grad():  # 禁用梯度计算，加快推理速度并节省内存
        for batch_idx, (inputs, _) in enumerate(data_loader):
            inputs = inputs.to(device)
            # 通过全局生成器生成特征图
            outputs = global_model(inputs)
            real_samples.append(feature_map.cpu())

    # 将所有批次的生成结果合并为一个张量
    real_samples = torch.cat(real_samples, dim=0)
    return real_samples


def local_train_net(nets, selected, args, net_dataidx_map, D, adv=False, test_dl=None, device="cpu"):
    avg_acc = 0.0

    G_output_list_all_clients = None
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        if not adv:
            trainacc, testacc, G_output_list = train_net_fedgan(net_id, net, train_dl_local, test_dl, n_epoch, args.lr,
                                                         args.optimizer, device=device)
        else:
            trainacc, testacc, G_output_list = adv_train_net(net_id, net, D, args.lambda_adv, train_dl_local, test_dl,
                                                             args.epoch_G, args.lr_G, args.optimizer_G, device=device)

        if G_output_list_all_clients is None:
            G_output_list_all_clients = G_output_list
        else:
            G_output_list_all_clients = torch.cat((G_output_list_all_clients, G_output_list), dim=0)
        logger.info('>> Shape of G_output_list_all_clients: {}'.format(G_output_list_all_clients.shape))

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, G_output_list_all_clients


def local_train_net_fedavg(nets, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedavg(net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer,
                                             device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc = train_net_fedprox(net_id, net, global_model, train_dl_local, test_dl, n_epoch, args.lr,
                                              args.optimizer, args.mu, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, test_dl=None,
                             device="cpu"):
    avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc, c_delta_para = train_net_scaffold(net_id, net, global_model, c_nets[net_id], c_global,
                                                             train_dl_local, test_dl, n_epoch, args.lr, args.optimizer,
                                                             device=device)

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]

        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
    for key in total_delta:
        total_delta[key] /= args.n_parties
    c_global_para = c_global.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            # print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


def local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map, test_dl=None, device="cpu"):
    avg_acc = 0.0

    a_list = []
    d_list = []
    n_list = []
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        trainacc, testacc, a_i, d_i = train_net_fednova(net_id, net, global_model, train_dl_local, test_dl, n_epoch,
                                                        args.lr, args.optimizer, device=device)

        a_list.append(a_i)
        d_list.append(d_i)
        n_i = len(train_dl_local.dataset)
        n_list.append(n_i)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list, a_list, d_list, n_list


def local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=None, global_model=None, prev_model_pool=None,
                         round=None, device="cpu"):
    avg_acc = 0.0
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        net.to(device)

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level, net_id, args.n_parties - 1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32,
                                                                 dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs

        prev_models = []
        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        trainacc, testacc = train_net_moon(net_id, net, global_model, prev_models, train_dl_local, test_dl, n_epoch,
                                           args.lr,
                                           args.optimizer, args.mu, args.temperature, args, round, device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc

    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)
    global_model.to('cpu')
    nets_list = list(nets.values())
    return nets_list


def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
    seed = init_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        dataset, datadir, logdir, partition, n_parties, beta=beta)

    return net_dataidx_map


if __name__ == '__main__':
    # torch.set_printoptions(profile="full")
    args = get_args()
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    # if args.log_file_name is None:
    #     argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    # else:
    #     argument_path=args.log_file_name+'.json'
    # with open(os.path.join(args.logdir, argument_path), 'w') as f:
    #     json.dump(str(args), f)
    device = torch.device(args.device)
    # logging.basicConfig(filename='test.log', level=logger.info, filemode='w')
    # logging.info("test")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(str(args))
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    logger.info("Partitioning data")
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)

    print("len train_dl_global:", len(train_ds_global))

    data_size = len(test_ds_global)

    # test_dl = data.DataLoader(dataset=test_ds_global, batch_size=32, shuffle=False)

    train_all_in_list = []
    test_all_in_list = []
    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size, 32,
                                                                                              dataidxs, noise_level,
                                                                                              party_id,
                                                                                              args.n_parties - 1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset,
                                                                                              args.datadir,
                                                                                              args.batch_size, 32,
                                                                                              dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)
        logger.info("Getting global dataset using ConcatDataset.")
    else:
        train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                          args.datadir,
                                                                                          args.batch_size,
                                                                                          32)
        logger.info("Getting global dataset using get_dataloader.")

    # 现在我们强制使用比较纯净（只加了noise，我也说不清）的数据集，虽然不确定会带来什么影响……
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)

    # 附加D
    # 创建判别器模型
    # 附加D
    # 创建判别器模型
    if args.dataset == 'mnist':
        if args.model == 'vgg9':
            D = DiscriminatorS_vgg9_mnist()
        elif args.model == 'simple-cnn':
            D = DiscriminatorS_simplecnn_mnist()
    elif args.dataset == 'cifar10':
        if args.model == 'resnet18':
            D = DiscriminatorS()  #              输入通道可以根据数据调整，例如灰度图使用 input_channels=1
        elif args.model == 'resnet50':
            D = DiscriminatorS_resnet50_cifar10()
        elif args.model == 'vgg11':
            D = Discriminator_vgg11_cifar10()
        elif args.model == 'simple-cnn':
            D = DiscriminatorS_simplecnn_cifar10()  # 输入通道可以根据数据调整，例如灰度图使用 input_channels=1
        elif args.model == 'vgg9':
            D = DiscriminatorS_vgg9_cifar10()
    elif args.dataset == 'fmnist':
        if args.model == 'simple-cnn':
            D = DiscriminatorS_simplecnn_mnist()
        elif args.model == 'vgg9':
            D = DiscriminatorS_vgg9_mnist()
    elif args.dataset == 'femnist':
        if args.model == 'simple-cnn':
            D = DiscriminatorS_simplecnn_mnist()
        elif args.model == 'vgg9':
            D = DiscriminatorS_simplecnn_mnist()

    D = D.to(device)
    print(D)

    # 初始化优化器和损失函数
    d_learning_rate = 0.01
    loss_d = nn.BCELoss()  # - [p * log(q) + (1-p) * log(1-q)]
    optimiser_D = optim.Adam(D.parameters(), lr=d_learning_rate)

    if args.alg == 'fedgan':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]
        print(nets[0])                                                                                                                                             
        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            # update_client_task_layers(global_model, nets)

            # if round == 0:
            #     _, G_output_list_all_clients = local_train_net(nets, selected, args, net_dataidx_map, D, adv = False, test_dl = test_dl_global, device=device)
            # else:
            #     _, G_output_list_all_clients = local_train_net(nets, selected, args, net_dataidx_map, D, adv = True, test_dl = test_dl_global, device=device)
            # # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            _, G_output_list_all_clients = local_train_net(nets, selected, args, net_dataidx_map, D, adv=False,
                                                           test_dl=test_dl_global, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            # ================第二轮 对抗训练===========================

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)
            # update_client_task_layers(global_model, nets)

            # 第二轮 对抗训练
            global_model.eval()
            real_data = generate_real_samples(global_model, train_dl_global, device=device)
            logger.info('>> Shape of generated real samples: ' + str(real_data.shape))

            # 判别器训练

            # TODO: 获取feature map，把对客户端的遍历放进函数里面
            G_output_list_all_clients = get_feature_maps(nets, selected, args, net_dataidx_map, test_dl=test_dl_global,
                                                         device=device)

            train_discriminator(D, G_output_list_all_clients, real_data, net_dataidx_map, device, args.optimizer_D,
                                args.lr_D, args.epoch_D,
                                batch_size=args.batch_size)

            # 对抗训练客户端
            # 将判别器的输出形成的loss发送到各client，让它们进行对抗训练
            # 这里首先进了local_train_net函数中，然后再分流到执行adv_train_net函数
            # local_train_net(nets, selected, args, net_dataidx_map, D, adv=True, test_dl=test_dl_global, device=device)
            _, G_output_list_all_clients = local_train_net(nets, selected, args, net_dataidx_map, D, adv=True,
                                                           test_dl=test_dl_global, device=device)

            # 第二轮update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            # aggregate_task_layers_weighted(global_model, nets, net_dataidx_map, selected)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            # 保存global model
            # Save the training state
            checkpoint = {
                'model': global_model,
                'round': round
            }
            filename = f"global_round{round}.pth"
            save_global_model(checkpoint, args.ckpt_dir, filename)

    elif args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedavg(nets, selected, args, net_dataidx_map, test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fedprox':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_fedprox(nets, selected, global_model, args, net_dataidx_map, test_dl=test_dl_global,
                                    device=device)
            global_model.to('cpu')

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'scaffold':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        c_globals, _, _ = init_nets(args.net_config, 0, 1, args)
        c_global = c_globals[0]
        c_global_para = c_global.state_dict()
        for net_id, net in c_nets.items():
            net.load_state_dict(c_global_para)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,
                                     test_dl=test_dl_global, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'fednova':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        d_list = [copy.deepcopy(global_model.state_dict()) for i in range(args.n_parties)]
        d_total_round = copy.deepcopy(global_model.state_dict())
        for i in range(args.n_parties):
            for key in d_list[i]:
                d_list[i][key] = 0
        for key in d_total_round:
            d_total_round[key] = 0

        data_sum = 0
        for i in range(args.n_parties):
            data_sum += len(traindata_cls_counts[i])
        portion = []
        for i in range(args.n_parties):
            portion.append(len(traindata_cls_counts[i]) / data_sum)

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            _, a_list, d_list, n_list = local_train_net_fednova(nets, selected, global_model, args, net_dataidx_map,
                                                                test_dl=test_dl_global, device=device)
            total_n = sum(n_list)
            # print("total_n:", total_n)
            d_total_round = copy.deepcopy(global_model.state_dict())
            for key in d_total_round:
                d_total_round[key] = 0.0

            for i in range(len(selected)):
                d_para = d_list[i]
                for key in d_para:
                    # if d_total_round[key].type == 'torch.LongTensor':
                    #    d_total_round[key] += (d_para[key] * n_list[i] / total_n).type(torch.LongTensor)
                    # else:
                    d_total_round[key] += d_para[key] * n_list[i] / total_n

            # for i in range(len(selected)):
            #     d_total_round = d_total_round + d_list[i] * n_list[i] / total_n

            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            coeff = 0.0
            for i in range(len(selected)):
                coeff = coeff + a_list[i] * n_list[i] / total_n

            updated_model = global_model.state_dict()
            for key in updated_model:
                # print(updated_model[key])
                if updated_model[key].type() == 'torch.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                elif updated_model[key].type() == 'torch.cuda.LongTensor':
                    updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                else:
                    # print(updated_model[key].type())
                    # print((coeff*d_total_round[key].type()))
                    updated_model[key] -= coeff * d_total_round[key]
            global_model.load_state_dict(updated_model)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

    elif args.alg == 'moon':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        old_nets_pool = []
        old_nets = copy.deepcopy(nets)
        for _, net in old_nets.items():
            net.eval()
            for param in net.parameters():
                param.requires_grad = False

        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)

            local_train_net_moon(nets, selected, args, net_dataidx_map, test_dl=test_dl_global,
                                 global_model=global_model,
                                 prev_model_pool=old_nets_pool, round=round, device=device)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):
                net_para = nets[selected[idx]].cpu().state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.load_state_dict(global_para)

            logger.info('global n_training: %d' % len(train_dl_global))
            logger.info('global n_test: %d' % len(test_dl_global))

            global_model.to(device)
            train_acc = compute_accuracy(global_model, train_dl_global, moon_model=True, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     moon_model=True, device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            if len(old_nets_pool) < 1:
                old_nets_pool.append(old_nets)
            else:
                old_nets_pool[0] = old_nets

    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        arr = np.arange(args.n_parties)
        local_train_net(nets, arr, args, net_dataidx_map, test_dl=test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, 1, args)
        n_epoch = args.epochs
        nets[0].to(device)
        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer,
                                      device=device)

        logger.info("All in test acc: %f" % testacc)


