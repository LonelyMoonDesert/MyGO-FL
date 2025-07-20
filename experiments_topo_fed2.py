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
import matplotlib.pyplot as plt
import umap
import datetime
from matplotlib import colors
from model import *
from utils import *
from vggmodel import *
from resnetcifar import *
from pytorch_pretrained_vit import ViT
from sklearn.neighbors import KernelDensity
from geomloss import SamplesLoss
from gudhi import CubicalComplex
from gudhi.representations import PersistenceImage
from gudhi import plot_persistence_barcode
from scipy.interpolate import make_interp_spline
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import gc
import psutil
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import gudhi
import torch.nn.functional as F
from torch import nn
import seaborn as sns

pi = PersistenceImage()  # 可以指定分辨率等参数
# 定义绘图颜色
custom_colors = ['#b2b1cf', '#eac7c7', '#e3d6b5']
# 举例：莫兰迪蓝渐变
morandi_blue_cmap = LinearSegmentedColormap.from_list(
    "morandi_blue", ["#b2b1cf", "#eaeaea", "#ffffff"]
)
# 全局字体
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "figure.titlesize": 18
})

# 自定义深蓝莫兰迪色系+反转
morandi_deepblue_cmap = LinearSegmentedColormap.from_list(
    "morandi_deepblue", ["#363c55", "#6b7a99", "#b2b1cf", "#eaeaea", "#ffffff"]
)
morandi_deepblue_cmap_r = morandi_deepblue_cmap.reversed()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp',
                        help='neural network used in training. resnet18/resnet50/vgg11/simple-cnn/')
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
    parser.add_argument('--n_parties', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='fedavg',
                        help='fl algorithms: fedavg/fedprox/scaffold/fednova/moon')
    parser.add_argument('--training_type', type=str, default='local', help='local/adversarial')
    parser.add_argument('--use_projection_head', type=bool, default=False,
                        help='whether add an additional header to model or not (see MOON)')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--loss', type=str, default='contrastive', help='for moon')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--is_same_initial', type=int, default=1,
                        help='Whether initial all the models with the same parameters in fedavg')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="../data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--l1_lambda', type=float, default=1e-4, help="L1 regularization strength")
    parser.add_argument('--l1', type=bool, default=False, help="Use L1 regularization")
    parser.add_argument('--adv_l1_lambda', type=float, default=1e-4, help="L1 regularization strength")
    parser.add_argument('--adv_l1', type=bool, default=False, help="Use L1 regularization")
    parser.add_argument('--lambda_adv', type=float, default=0.3, help="adv_loss")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./models/", help='Model directory path')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--optimizer_G', type=str, default='amsgrad', help='the optimizer')
    parser.add_argument('--optimizer_D', type=str, default='amsgrad', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.001, help='the mu parameter for fedprox')
    parser.add_argument('--noise', type=float, default=0, help='how much noise we add to some party')
    parser.add_argument('--noise_type', type=str, default='level',
                        help='Different level of noise or different space of noise')
    parser.add_argument('--rho', type=float, default=0, help='Parameter controlling the momentum SGD')
    parser.add_argument('--sample', type=float, default=1, help='Sample ratio for each communication round')

    # for scheduler
    parser.add_argument('--scheduler', type=str, default='StepLR', help='Type of scheduler: StepLR, MultiStepLR, etc.')
    parser.add_argument('--scheduler_g', type=str, default='StepLR',
                        help='Type of scheduler: StepLR, MultiStepLR, etc.')
    parser.add_argument('--scheduler_d', type=str, default='StepLR',
                        help='Type of scheduler: StepLR, MultiStepLR, etc.')
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
    # ‘conv1’-simplecnn, 'layer3'-resnet18
    parser.add_argument('--feature_layer', type=str, default='conv1', help='用于特征提取和拓扑分析的层名（如conv1/layer3等）')

    # UMAP可视化相关
    parser.add_argument('--max_samples', type=int, default=1000, help='最大样本数量')
    parser.add_argument('--umap_dim', type=int, default=3, help='UMAP降维后的维度')
    parser.add_argument('--n_umap_batches', type=int, default=4, help='UMAP可视化采样的batch数')

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
    elif args.dataset == 'imagenet':
        n_classes = 1000  # ImageNet has 1000 classes

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
                # For ImageNet dataset, choose from several popular pretrained models
                if args.dataset in ("imagenet", "tinyimagenet"):
                    if args.model == "resnet18":
                        net = models.resnet18(pretrained=True)
                    elif args.model == "resnet50":
                        net = models.resnet50(pretrained=True)
                    elif args.model == "vgg16":
                        net = models.vgg16(pretrained=True)
                    elif args.model == "vit":
                        net = ViT('B_16_imagenet1k', pretrained=True)
                    else:
                        print("Model not supported for ImageNet")
                        exit(1)
                elif args.dataset == "generated":
                    net = PerceptronModel()
                elif args.model == "mlp":
                    if args.dataset == 'covtype':
                        input_size = 54
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                        net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                        hook_handle = net.layers[2].register_forward_hook(hook_fn)
                    elif args.dataset == 'a9a':
                        input_size = 123
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                        net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                        hook_handle = net.layers[2].register_forward_hook(hook_fn)
                    elif args.dataset == 'rcv1':
                        input_size = 47236
                        output_size = 2
                        hidden_sizes = [32, 16, 8]
                        net = FcNet(input_size, hidden_sizes, output_size, dropout_p)
                        hook_handle = net.layers[2].register_forward_hook(hook_fn)
                    elif args.dataset == 'SUSY':
                        input_size = 18
                        output_size = 2
                        hidden_sizes = [16, 8]
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
                    if args.dataset in ("cifar10", "cinic10", "svhn"):
                        net = ResNet18_cifar10()
                        hook_handle = net.layer3.register_forward_hook(hook_fn_reduce_feature_map)
                    elif args.dataset in ("tinyimagenet", "imagenet"):
                        net = ResNet18_cifar10()
                        hook_handle = net.layer3.register_forward_hook(hook_fn_reduce_feature_map)
                elif args.model == "vgg16":
                    net = vgg16()
                elif args.model == "vit":
                    net = ViT('B_16_imagenet1k', pretrained=True)

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
                    # features = features.cpu()  # Ensure features is on CPU
                    if G_output_list is None:
                        G_output_list = features  # Initialize with the first feature map
                    else:
                        # Concatenate along the batch dimension
                        G_output_list = torch.cat((G_output_list, features), dim=0)

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
    # criterion_adv = nn.BCELoss().to(device)
    criterion_adv = nn.BCEWithLogitsLoss().to(device)

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    # 初始化优化器，只对生成器部分的参数进行优化
    # 获取生成器部分的参数（conv1 到 layer3 之前的所有层）
    generator_parameters = []
    G_output_list = None

    # for mlp
    if args.model == "mlp":
        # 获取FcNet中的参数，选择layer0, layer1, layer2的参数
        for name, param in net.named_parameters():
            if 'layers.3' in name:  # 将layer3及以后的层设置为不更新
                param.requires_grad = False
                # print(name + " not updated.")
            else:  # 将layer0, layer1, layer2的参数加入generator_parameters
                generator_parameters.append(param)
    else:
        for name, param in net.named_parameters():
            if 'layer4' in name or 'avgpool' in name or 'fc' in name:
                param.requires_grad = False
                # print(name+" added to parameters.")
            else:
                # print(name + " added to parameters.")
                generator_parameters.append(param)

    # print(net.state_dict())
    optimizer = optim.SGD(generator_parameters, lr=lr, momentum=args.rho, weight_decay=args.reg)
    # print(generator_parameters)

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
                # fake_labels = torch.full((x.shape[0],), 0.1, device=device)
                # real_labels = torch.full((x.shape[0],), 0.9, device=device)
                real_labels = torch.full((x.shape[0],), 0.85, device=device)
                fake_labels = torch.full((x.shape[0],), 0.15, device=device)

                # 前向传播，计算生成器输出
                out = net(x)  # 通过模型的前半部分

                # 在最后一个 epoch 记录下 G_output
                if epoch == epochs - 1:
                    # features = features.cpu()  # Ensure features is on CPU
                    if G_output_list is None:
                        G_output_list = features  # Initialize with the first feature map
                    else:
                        # Concatenate along the batch dimension
                        G_output_list = torch.cat((G_output_list, features), dim=0)

                # 获取生成器的中间特征图 (通过 hook 获得 features)
                D_out = D(features).squeeze()
                # print('D_out: {}'.format(D_out))
                # print('fake_labels: {}'.format(fake_labels))
                # epsilon = 1e-7
                # D_out = D_out.clamp(min=epsilon, max=1 - epsilon)
                D_out = D_out.view(-1, 1)  # Ensure D_out shape is (batch_size, 1)
                real_labels = real_labels.view(-1, 1)  # Ensure real_labels shape is (batch_size, 1)
                # 计算任务损失和对抗损失
                task_loss = criterion_task(out, target)
                # adv_loss = criterion_adv(D_out, fake_labels)
                adv_loss = criterion_adv(D_out, real_labels)

                # print("out: ", out)
                # print("target: ", target)
                # print("D_out: ", D_out)
                # print("real_labels: ", real_labels)
                # 计算总损失
                l1_norm = sum(p.abs().sum() for p in net.parameters())

                # 计算总损失
                if args.model == "mlp":
                    if args.l1:
                        loss = task_loss + adv_loss + 0.01 * args.l1_lambda * l1_norm
                    else:
                        loss = task_loss + adv_loss
                else:
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
                #
                # # 在训练结束后计算并记录特征的熵
                # logger.info('Post-training entropy of features: {}'.format(calculate_entropy(net(x))))

        # 记录每个epoch最后一个batch的损失
        # logger.info('Epoch: %d Last Batch Total Loss: %f Task Loss: %f Adversarial Loss: %f' % (
        # epoch, epoch_loss_collector[-1], epoch_task_loss_collector[-1], epoch_adv_loss_collector[-1]))

        # 记录每个epoch最后一个batch的损失
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_task_loss = sum(epoch_task_loss_collector) / len(epoch_task_loss_collector)
        epoch_adv_loss = sum(epoch_adv_loss_collector) / len(epoch_adv_loss_collector)

        if args.l1:
            epoch_l1_loss = sum(epoch_l1_loss_collector) / len(epoch_l1_loss_collector)
            logger.info('Epoch: %d Last Batch Total Loss: %f Task Loss: %f Adversarial Loss: %f L1 norm: %f' % (
            epoch, epoch_loss, epoch_task_loss, epoch_adv_loss, epoch_l1_loss))
        else:
            logger.info('Epoch: %d Last Batch Total Loss: %f Task Loss: %f Adversarial Loss: %f ' % (
            epoch, epoch_loss, epoch_task_loss, epoch_adv_loss))

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
    # print(net.state_dict())
    filename = f"adv_client{net_id}_round{round}.pth"
    save_checkpoint(checkpoint, './checkpoints', filename)

    # 记录这次的平均熵值
    entropy = compute_entropy(G_output_list)

    # 记录这次的平均熵值
    logger.info('>> Entropy: %f' % entropy)

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
    sample_idx = 0  # 用于记录 features 的样本位置
    total_samples = len(train_dataloader[0].dataset)  # 总样本数（假设数据集大小已知）
    feature_map_shape = (total_samples, 64, 4, 4)  # 假设每个 feature map 的尺寸是 [64, 4, 4]
    G_output_list = torch.zeros(feature_map_shape, device=device)

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

                out = net(x)  # 假设返回的 features 是你想记录的

                # 仅在最后一个 epoch 时记录 G_output
                if epoch == epochs - 1:
                    G_output_list[sample_idx:sample_idx + len(x)] = features.detach()
                    sample_idx += len(x)

                if args.l1:
                    l1_norm = sum(p.abs().sum() for p in net.parameters())
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

    entropy = compute_entropy(G_output_list)

    # 记录这次的平均熵值
    logger.info('>> Entropy: %f' % entropy)

    return train_acc, test_acc, G_output_list

def train_net_fedtopo(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer,
                      global_model, history, round, device="cpu", pi=None, K=8, pool_size=32, alpha=0.3):
    """
    FedTopo 训练核心新实现！彻底避免特征“短路”，使用PI signature欧氏loss。
    pi: gudhi.representations.PersistenceImage 实例, K: PCA分量数, alpha: topo loss权重
    """

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho,
                              weight_decay=args.reg)
    criterion = nn.CrossEntropyLoss().to(device)

    logger.info('Training network %s' % str(net_id))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))

    if type(train_dataloader) != type([1]):
        train_dataloader = [train_dataloader]

    for epoch in range(epochs):
        total_loss_collector, ce_loss_collector, topo_loss_collector = [], [], []
        last_local_pi, last_global_pi, last_out = None, None, None
        for tmp in train_dataloader:
            for x, target in tmp:
                x, target = x.to(device), target.to(device)
                optimizer.zero_grad()
                x.requires_grad = False
                target.requires_grad = False
                target = target.long()

                # === 1. forward 得到 local/global PI 向量 ===
                local_feat  = extract_layer_features(net, x,  layer_name=args.feature_layer, pool_size=pool_size, device=device)
                global_feat = extract_layer_features(global_model, x, layer_name=args.feature_layer, pool_size=pool_size, device=device)
                local_pi  = batch_channel_pi(local_feat, K=K, pi=pi)   # [B, M]
                global_pi = batch_channel_pi(global_feat, K=K, pi=pi)  # [B, M]

                # === 2. Loss ===
                out = net(x)
                ce_loss = criterion(out, target)
                topo_loss = torch.norm(local_pi - global_pi, p=2) / x.shape[0]
                total_loss = ce_loss + alpha * topo_loss

                total_loss.backward()
                optimizer.step()

                total_loss_collector.append(total_loss.item())
                ce_loss_collector.append(ce_loss.item())
                topo_loss_collector.append(topo_loss.item())

                # 只记录最后一个batch的特征和输出
                last_local_pi = local_pi.detach()
                last_global_pi = global_pi.detach()
                last_out = out.detach()

        epoch_total_loss = np.mean(total_loss_collector)
        epoch_ce_loss = np.mean(ce_loss_collector)
        epoch_topo_loss = np.mean(topo_loss_collector)
        logger.info(f'Epoch: {epoch} Total: {epoch_total_loss:.4f} CE: {epoch_ce_loss:.4f} Topo: {epoch_topo_loss:.4f}')

    # 结尾照常统计 acc/记录 history ...
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)
    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')

    # === 用最后一个batch的local_pi/global_pi和out统计指标 ===
    # 1. 特征相似性（以欧氏距离为例，或者你可以换成别的指标）
    if last_local_pi is not None and last_global_pi is not None:
        similarity = torch.nn.functional.cosine_similarity(
            last_local_pi.flatten(1), last_global_pi.flatten(1)
        ).mean().item()
        distance = torch.norm(last_local_pi - last_global_pi, p=2).item() / last_local_pi.shape[0]
    else:
        similarity = 0.0
        distance = 0.0
    logger.info(f">> Last-batch PI Similarity: {similarity:.4f}")
    logger.info(f">> Last-batch Topo Distance: {distance:.4f}")
    cka_val = cka(local_feat, global_feat)
    history['client_cka'][net_id][round] = 1 - cka_val  # 越小越相似
    logger.info(f'>> Last-batch 1-CKA: {1 - cka_val:.4f}')

    # 2. 用最后一个batch的logits算熵
    if last_out is not None:
        probs = torch.nn.functional.softmax(last_out, dim=1)
        log_probs = torch.log(probs + 1e-7)
        entropy = -torch.sum(probs * log_probs, dim=1).mean().item()
        logger.info('>> Last-batch Entropy: %f' % entropy)
    else:
        entropy = 0.0
    # 更新历史记录
    history['client_total_loss'][net_id][round] = epoch_total_loss
    history['client_ce_loss'][net_id][round] = epoch_ce_loss
    history['client_topo_loss'][net_id][round] = epoch_topo_loss
    history['client_train_acc'][net_id][round] = train_acc
    history['client_test_acc'][net_id][round] = test_acc
    history['client_topo_distance'][net_id][round] = distance
    history['client_similarity'][net_id][round] = similarity
    history['client_entropy'][net_id][round] = entropy
    delta = distance - history['client_topo_distance'][net_id][round - 1] if round > 0 else 0.0
    history['client_delta_topo_dist'][net_id][round] = delta
    dists = [history['client_topo_distance'][net_id][round] for net_id in range(args.n_parties)]
    history['round_var'][round] = float(np.var(dists))
    swd_val = swd(last_local_pi, last_global_pi)
    history['client_swd'][net_id][round] = swd_val
    logger.info(f'>> Last-batch SWD: {swd_val:.4f}')

    return train_acc, test_acc


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
    G_output_list = None
    for epoch in range(epochs):
        total_loss_collector = []
        last_local_pi, last_global_pi, last_out = None, None, None
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                # === 1. forward 得到 local/global PI 向量 ===
                # TODO:待修改
                local_feat  = extract_layer_features(net, x,  layer_name=args.feature_layer, pool_size=pool_size, device=device)
                global_feat = extract_layer_features(global_model, x, layer_name=args.feature_layer, pool_size=pool_size, device=device)
                local_pi  = batch_channel_pi(local_feat, K=K, pi=pi)   # [B, M]
                global_pi = batch_channel_pi(global_feat, K=K, pi=pi)  # [B, M]

                out = net(x)

                # 在最后一个 epoch 记录下 G_output
                if epoch == epochs - 1:
                    # features = features.cpu()  # Ensure features is on CPU
                    if G_output_list is None:
                        G_output_list = features  # Initialize with the first feature map
                    else:
                        # Concatenate along the batch dimension
                        G_output_list = torch.cat((G_output_list, features), dim=0)

                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                total_loss_collector.append(loss.item())

                # 只记录最后一个batch的特征和输出
                last_local_pi = local_pi.detach()
                last_global_pi = global_pi.detach()
                last_out = out.detach()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_total_loss = np.mean(total_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_total_loss))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)

    net.to('cpu')
    logger.info(' ** Training complete **')

    # === 用最后一个batch的local_pi/global_pi和out统计指标 ===
    # 1. 特征相似性（以欧氏距离为例，或者你可以换成别的指标）
    if last_local_pi is not None and last_global_pi is not None:
        similarity = torch.nn.functional.cosine_similarity(
            last_local_pi.flatten(1), last_global_pi.flatten(1)
        ).mean().item()
        distance = torch.norm(last_local_pi - last_global_pi, p=2).item() / last_local_pi.shape[0]
    else:
        similarity = 0.0
        distance = 0.0
    logger.info(f">> Last-batch PI Similarity: {similarity:.4f}")
    logger.info(f">> Last-batch Topo Distance: {distance:.4f}")
    cka_val = cka(local_feat, global_feat)
    history['client_cka'][net_id][round] = 1 - cka_val  # 越小越相似
    logger.info(f'>> Last-batch 1-CKA: {1 - cka_val:.4f}')

    # 2. 用最后一个batch的logits算熵
    if last_out is not None:
        probs = torch.nn.functional.softmax(last_out, dim=1)
        log_probs = torch.log(probs + 1e-7)
        entropy = -torch.sum(probs * log_probs, dim=1).mean().item()
        logger.info('>> Last-batch Entropy: %f' % entropy)
    else:
        entropy = 0.0
    # 更新历史记录
    history['client_total_loss'][net_id][round] = epoch_total_loss
    history['client_ce_loss'][net_id][round] = epoch_total_loss
    history['client_topo_loss'][net_id][round] = 0.0
    history['client_train_acc'][net_id][round] = train_acc
    history['client_test_acc'][net_id][round] = test_acc
    history['client_topo_distance'][net_id][round] = distance
    history['client_similarity'][net_id][round] = similarity
    history['client_entropy'][net_id][round] = entropy
    delta = distance - history['client_topo_distance'][net_id][round - 1] if round > 0 else 0.0
    history['client_delta_topo_dist'][net_id][round] = delta
    dists = [history['client_topo_distance'][net_id][round] for net_id in range(args.n_parties)]
    history['round_var'][round] = float(np.var(dists))
    swd_val = swd(last_local_pi, last_global_pi)
    history['client_swd'][net_id][round] = swd_val
    logger.info(f'>> Last-batch SWD: {swd_val:.4f}')

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


features = None


# 定义 hook 函数
def hook_fn(module, input, output):
    global features
    features = output  # 将中间输出存储到全局变量中


def hook_fn_to_2d(module, input, output):
    global features
    # 全局平均池化
    pooled_output = nn.functional.adaptive_avg_pool2d(output, (1, 1))
    # 展平为二维，形状为 (batch_size, channels)
    flattened_output = pooled_output.view(output.size(0), -1)
    # 使用线性层降维，将通道数从 256 降到 2
    linear = nn.Linear(256, 2).to(output.device)
    features = linear(flattened_output)


# 定义 hook 函数，进行特征图降维
def hook_fn_reduce_feature_map(module, input, output):
    global features
    # 使用卷积降维，将通道数从 256 降到 m，空间维度从 (8, 8) 到 (n, n)
    m, n = 64, 4  # 可以根据需要调整 m 和 n 的值
    # 卷积层用于减少通道数
    conv = nn.Conv2d(in_channels=256, out_channels=m, kernel_size=3, stride=2, padding=1).to(output.device)
    reduced_output = conv(output)
    # 使用自适应平均池化调整空间维度为 (n, n)
    reduced_output = nn.functional.adaptive_avg_pool2d(reduced_output, (n, n))
    features = reduced_output


# 定义 hook 函数，resnet50-cifar10
def hook_fn_resnet50_cifar10(module, input, output):
    global features
    # 使用卷积降维，将通道数从 1024 降到 m，空间维度从 (8, 8) 到 (n, n)
    m, n = 64, 4  # 可以根据需要调整 m 和 n 的值
    # 卷积层用于减少通道数，这里将 in_channels 设置为 1024，匹配输入的通道数
    conv = nn.Conv2d(in_channels=1024, out_channels=m, kernel_size=3, stride=2, padding=1).to(output.device)

    # 先用卷积降维通道数
    reduced_output = conv(output)

    # 使用自适应平均池化调整空间维度为 (n, n)
    reduced_output = nn.functional.adaptive_avg_pool2d(reduced_output, (n, n))

    # 保存特征图
    features = reduced_output

# 收集全局特征的函数
def collect_global_features(global_model, train_dl_global, device):
    global_samples = []
    global_targets = []
    global_model.to(device)
    with torch.no_grad():
        for data, target in train_dl_global:
            data = data.to(device)
            out = global_model(data)
            global_features = features
            global_features = global_features.view(global_features.size(0), -1)
            global_samples.append(global_features.cpu())
            global_targets.append(target.numpy())  # 收集标签
    return torch.cat(global_samples, dim=0).to(device), np.concatenate(global_targets, axis=0)

# 计算熵的函数
def compute_entropy(features):
    # 将每个特征图展平
    flattened = features.view(features.size(0), -1)

    # 归一化每个特征图成概率分布（在每个batch的维度上应用softmax）
    probs = F.softmax(flattened, dim=1)

    # 计算熵
    log_probs = torch.log(probs + 1e-7)  # 加上一个小的epsilon避免log(0)
    entropy = -torch.sum(probs * log_probs, dim=1)  # 计算每个特征图的熵

    # 返回所有特征图的熵的平均值
    return torch.mean(entropy)

def compute_feature_similarity(client, global_):
    # 计算余弦相似度
    client_mean = torch.mean(normalize_features(client), dim=0)
    global_mean = torch.mean(normalize_features(global_), dim=0)
    return torch.cosine_similarity(client_mean.unsqueeze(0), global_mean.unsqueeze(0))

# 计算全局模型在全局数据集上的熵
def compute_global_entropy(global_model, train_dataloader, device):
    global_model.eval()  # 切换到评估模式
    all_feature_maps = []

    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)

            # 获取模型输出（特征图）
            out = global_model(x)

            # 将每个批次的特征图收集到一个列表中
            all_feature_maps.append(features)

    # 将所有批次的输出合并成一个张量
    all_feature_maps = torch.cat(all_feature_maps, dim=0)

    # 计算并返回熵
    return compute_entropy(all_feature_maps)

def normalize_features(features):
    return features / (torch.norm(features, dim=1, keepdim=True) + 1e-6)


def compute_feature_alignment_loss(client_features, global_features, scaling_factor=1.0):
    """
    计算客户端特征和全局特征的统计对齐损失（均值和协方差），并引入一个缩放因子来控制权重
    """
    # 计算均值差异
    client_features = normalize_features(client_features)
    global_features = normalize_features(global_features)
    client_mean = torch.mean(client_features, dim=0)
    global_mean = torch.mean(global_features, dim=0)
    mean_loss = torch.norm(client_mean - global_mean, p=2)

    # 计算协方差差异（添加小量防止奇异矩阵）
    client_cov = torch.cov(client_features.T + 1e-4 * torch.randn_like(client_features.T))
    global_cov = torch.cov(global_features.T + 1e-4 * torch.randn_like(global_features.T))
    cov_loss = torch.norm(client_cov - global_cov, p='fro')

    return scaling_factor * (mean_loss + 0.1 * cov_loss)


def compute_geometric_alignment_loss(client_features, global_features, scaling_factor=1.0, max_samples=256):
    """
    使用Wasserstein距离衡量分布几何差异（带采样）
    """
    # 随机采样
    idx_client = torch.randperm(client_features.size(0))[:max_samples]
    idx_global = torch.randperm(global_features.size(0))[:max_samples]

    client_sample = normalize_features(client_features[idx_client])
    global_sample = normalize_features(global_features[idx_global])

    # 计算Wasserstein距离
    wasserstein_loss = SamplesLoss(
        loss="sinkhorn",
        p=2,
        blur=0.05,
        scaling=0.8,  # 增加缩放系数加速收敛
        debias=False  # 关闭去偏置减少计算量
    )(client_sample, global_sample)

    return scaling_factor * wasserstein_loss

def hybrid_alignment_loss(client_features, global_features, alpha=0.9):
    statistical_loss = compute_feature_alignment_loss(client_features, global_features)  # 原统计损失
    geometric_loss = compute_geometric_alignment_loss(client_features, global_features)  # 新几何损失
    return alpha * geometric_loss + (1 - alpha) * statistical_loss

def extract_layer_features(model, x, layer_name='layer3', pool_size=32, device='cuda'):
    """
    获取某模型某一层的输出，并池化到 pool_size×pool_size。
    返回: tensor, shape [B, C, pool_size, pool_size]
    """
    feats = []
    def hook_fn(module, input, output):
        feats.append(output.detach())
    # 找到目标层
    layer = dict([*model.named_modules()])[layer_name]
    handle = layer.register_forward_hook(hook_fn)
    _ = model(x.to(device))
    handle.remove()
    feat = feats[0]
    feat = nn.functional.adaptive_avg_pool2d(feat, pool_size)
    # print('pooled feat', feat.shape)    # pooled feat torch.Size([32, 256, 8, 8])
    # print('first image, first channel', feat[0, 0])
    # print('second image, first channel', feat[1, 0])
    # print('mean diff', (feat[0] - feat[1]).abs().mean())
    return feat

def batch_channel_pi(feat_bchw, K=8, pi=None):
    B, C, H, W = feat_bchw.shape
    pi_vecs = []
    feat_cpu = feat_bchw.cpu().detach()
    for i in range(B):
        vlist = []
        for k in range(min(K, C)):
            # 检查arr内容
            arr = feat_cpu[i, k].numpy()
            # print(f"batch {i}, channel {k}, arr mean/std:", arr.mean(), arr.std(), "max/min", arr.max(), arr.min())
            cc = gudhi.CubicalComplex(dimensions=arr.shape, top_dimensional_cells=arr.ravel())
            bars = cc.persistence()
            # 只取条形码 birth-death 二元组
            bars_pd = [pair[1] for pair in bars if len(pair) > 1 and pair[1][1] > pair[1][0]]
            if len(bars_pd) == 0:
                bars_pd = np.zeros((0, 2), dtype=np.float32)  # 空 shape
            else:
                bars_pd = np.array(bars_pd, dtype=np.float32).reshape(-1, 2)
            try:
                v = pi.transform([bars_pd]).ravel()
            except Exception:
                # 兼容 PI transform 的各版本行为
                v = np.zeros(pi.fit([np.array([[0, 0]])]).transform([np.array([[0, 0]])]).shape[1], dtype=np.float32)
            vlist.append(v)
        vlist = np.stack(vlist, 0)
        pi_vecs.append(vlist.mean(axis=0))
    pi_vecs = torch.tensor(np.stack(pi_vecs, 0), device=feat_bchw.device, dtype=torch.float32)
    return pi_vecs

def fit_persistence_image_from_loader(model, dataloader, device, layer_name='layer3', pool_size=8, K=2, max_batches=20):
    from gudhi.representations import PersistenceImage
    pi = PersistenceImage()
    all_barcodes = []
    batch_count = 0

    for x, _ in dataloader:
        x = x.to(device)
        with torch.no_grad():
            feat = extract_layer_features(model, x, layer_name=layer_name, pool_size=pool_size, device=device)
        B, C, H, W = feat.shape
        feat_cpu = feat.cpu().detach()
        for i in range(B):
            for k in range(min(K, C)):
                arr = feat_cpu[i, k].numpy()
                cc = gudhi.CubicalComplex(dimensions=arr.shape, top_dimensional_cells=arr.ravel())
                bars = cc.persistence()
                bars_pd = [pair[1] for pair in bars if len(pair) > 1 and pair[1][1] > pair[1][0]]
                if len(bars_pd) == 0:
                    bars_pd = np.zeros((0, 2), dtype=np.float32)
                else:
                    bars_pd = np.array(bars_pd, dtype=np.float32).reshape(-1, 2)
                all_barcodes.append(bars_pd)
        batch_count += 1
        if batch_count >= max_batches:
            break

    # ----------- 关键：只保留有限数 ----------
    valid_barcodes = []
    for bars in all_barcodes:
        if bars.shape[0] == 0:
            continue
        finite_mask = np.isfinite(bars).all(axis=1)
        bars = bars[finite_mask]
        # 你可以顺便过滤 death-birth<1e-7 这样的空条
        nonzero_mask = (np.abs(bars[:, 1] - bars[:, 0]) > 1e-7)
        bars = bars[nonzero_mask]
        if bars.shape[0] > 0:
            valid_barcodes.append(bars)
    if len(valid_barcodes) == 0:
        valid_barcodes.append(np.array([[0, 0]], dtype=np.float32))

    pi.fit(valid_barcodes)
    return pi

def cka(X, Y):
    X = X.flatten(1); Y = Y.flatten(1)
    hsic = (X @ Y.T).pow(2).mean()
    norm = (X @ X.T).pow(2).mean().sqrt() * (Y @ Y.T).pow(2).mean().sqrt()
    return (hsic / (norm + 1e-8)).item()

def swd(u, v, n_proj=64):
    d = u.numel()
    u = u.flatten(); v = v.flatten()
    proj = torch.randn(d, n_proj, device=u.device)
    proj /= (proj.norm(dim=0, keepdim=True) + 1e-8)
    u_proj = torch.sort(u @ proj)[0]
    v_proj = torch.sort(v @ proj)[0]
    return torch.mean((u_proj - v_proj).abs()).item()

# 颜色与客户端标签
CLIENT_COLORS = ['#9370DB', '#6495ED', '#90EE90', '#F0E68C', '#F4A460']


def _smooth_curve(x, y, n=200, k=3):
    """
    把离散折线平滑成曲线。
    x / y 一维等长数组；n 为插值后的采样点数；k=3 为三次样条。
    """
    if len(x) < k + 1:          # 点太少则直接返回
        return x, y
    spline = make_interp_spline(x, y, k=min(k, len(x) - 1))
    x_smooth = np.linspace(x.min(), x.max(), n)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth


def plot_training_progress(history, n_clients, n_rounds):
    """
    绘制 FedTopo 训练过程 10 个关键指标（曲线版）。
    """
    # ---------- 布局 ----------
    fig, axs = plt.subplots(
        3, 4, figsize=(18, 12), dpi=300,           # ⬅︎ dpi=300，整体尺寸缩小
        constrained_layout=True                    # 类似 tight_layout，但更稳
    )
    xs = np.arange(1, n_rounds + 1)                # 1 ~ n_rounds
    round_labels = [str(r) for r in xs]

    # ---------- 每个客户端曲线 ----------
    for cid in range(n_clients):
        color = CLIENT_COLORS[cid % len(CLIENT_COLORS)]
        label = f'Client {cid + 1}'

        # 先把各指标数据取出来，统一平滑，再画
        for (r, c), key in [
            ((0, 0), 'client_total_loss'),
            ((0, 1), 'client_ce_loss'),
            ((0, 2), 'client_topo_loss'),
            ((0, 3), 'client_delta_topo_dist'),
            ((1, 0), 'client_train_acc'),
            ((1, 1), 'client_test_acc'),
            ((1, 2), 'client_topo_distance'),
            ((1, 3), 'client_similarity'),
            ((2, 0), 'client_entropy'),
            ((2, 2), 'client_cka'),
            ((2, 3), 'client_swd'),
        ]:
            y = history[key][cid][:n_rounds]
            x_s, y_s = _smooth_curve(xs, np.asarray(y))
            axs[r, c].plot(x_s, y_s, color=color,
                           linewidth=1.8,
                           label=label if (r, c) == (0, 0) else None)

    # ---------- 全局（单条黑线） ----------
    x_s, y_s = _smooth_curve(xs, np.asarray(history['round_var'][:n_rounds]))
    axs[2, 1].plot(x_s, y_s, color='black', linewidth=2.2)

    # ---------- 子图标题 ----------
    titles = [
        "Total Loss", "Cross‑Entropy Loss", "Topo Loss", "Δ‑Topo Distance",
        "Train Accuracy", "Test Accuracy", "Topo Distance", "Similarity (1‑CKA / Cos)",
        "Persistence Entropy", "Client CKA", "Client SWD", "Between‑Client Variance"
    ]
    for ax, title in zip(axs.flatten()[:len(titles)], titles):
        ax.set_title(title, fontsize=20)

    # ---------- 统一坐标轴 / 刻度 ----------
    for ax in axs.flatten():
        ax.set_xlabel("Communication Round", fontsize=18)
        ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)          # ⬅︎ 左右各留 0.5 的空白
        ax.set_xticks(xs)
        ax.set_xticklabels(round_labels, rotation=45, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.margins(x=0.03)                              # 细调额外边距

    # 只有第一张子图放 legend
    axs[0, 0].legend(loc='upper right', ncol=1, fontsize=16, frameon=True)

    return fig


# ----------- 可选：全局柔和风格 ----------
plt.style.use('seaborn-v0_8-muted')  # 柔和审美

def get_umap_sample_x(train_dl_global, n_batches=4, per_batch=32, device="cpu"):
    xs, ys, count = [], [], 0
    for xb, yb in train_dl_global:
        xs.append(xb[:per_batch])
        ys.append(yb[:per_batch])
        count += 1
        if count >= n_batches:
            break
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    return x.to(device), y.cpu().numpy()


# ========== 可视化相关函数 ==========
def compute_barcode_and_pi(feature_map, pi):
    # feature_map: numpy数组，单通道
    cc = CubicalComplex(dimensions=feature_map.shape, top_dimensional_cells=feature_map.ravel())
    bars = cc.persistence()
    bars_pd = [pair[1] for pair in bars if len(pair) > 1 and pair[1][1] > pair[1][0]]
    if len(bars_pd) == 0:
        bars_pd = np.zeros((0, 2), dtype=np.float32)
    else:
        bars_pd = np.array(bars_pd, dtype=np.float32).reshape(-1, 2)
    pi_vec = pi.transform([bars_pd]).reshape(pi.resolution)
    return bars, pi_vec

def plot_barcode(
    bars, ax, title="Barcode", linewidth=3,
    xlim=(0, 1.0), colors=None,
    title_fontsize=14, label_fontsize=13, legend_fontsize=12,
    show_legend=True  # 新增
):
    if colors is None:
        import matplotlib
        colors = matplotlib.colormaps['tab10'].colors[:3]
    labels = {0: 'H0 (Connected)', 1: 'H1 (Hole)', 2: 'H2 (Cavity)'}
    y = [0, 0, 0]
    legend_handles = []
    legend_labels = []
    for bar in bars:
        if len(bar) > 1:
            dim = bar[0]
            birth, death = bar[1]
            if death > birth:
                h = ax.hlines(y[dim], birth, death, color=colors[dim], linewidth=linewidth,
                          label=labels[dim] if y[dim] == 0 else "")
                if y[dim] == 0:  # 只收集第一个
                    legend_handles.append(h)
                    legend_labels.append(labels[dim])
                y[dim] += 1
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Birth/Death Value", fontsize=label_fontsize)
    ax.set_ylabel("Barcode Index", fontsize=label_fontsize)
    ax.set_yticks([])
    ax.set_xlim(xlim)
    if show_legend and legend_handles:
        ax.legend(legend_handles, legend_labels, fontsize=legend_fontsize, loc='best')
    return legend_handles, legend_labels



def plot_pi(
    pi_vec, ax, title="Persistence Image",
    cmap=None, vmin=None, vmax=None,
    title_fontsize=16,
    label_fontsize=15,
    colorbar_fontsize=13
):
    if cmap is None:
        # 默认莫兰迪色板+反转
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "morandi_deepblue", ["#363c55", "#6b7a99", "#b2b1cf", "#eaeaea", "#ffffff"]
        ).reversed()
    im = ax.imshow(pi_vec, cmap=cmap, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("Birth", fontsize=label_fontsize)
    ax.set_ylabel("Persistence", fontsize=label_fontsize)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)



def get_filtration_range(barcodes_list, margin=0.05, fallback=(0, 1)):
    births = []
    deaths = []
    for bars in barcodes_list:
        for bar in bars:
            if len(bar) > 1:
                birth, death = bar[1]
                if (death > birth) and (np.isfinite(birth)) and (np.isfinite(death)):
                    births.append(birth)
                    deaths.append(death)
    if not births or not deaths:
        return fallback
    min_birth = min(births)
    max_death = max(deaths)
    if not np.isfinite(min_birth) or not np.isfinite(max_death):
        return fallback
    span = max_death - min_birth
    if span == 0:
        min_birth -= 0.05
        max_death += 0.05
    else:
        min_birth -= margin * span
        max_death += margin * span
    # 最终再防一手
    if not np.isfinite(min_birth) or not np.isfinite(max_death):
        return fallback
    return (min_birth, max_death)


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
    Ensures that the directory structure is consislinewidth=2tent across various saving functions.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure the directory exists.
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print("Checkpoint saved to {}".format(filepath))

# 固定 10 个类别的颜色
CMAP = plt.get_cmap('tab10', 10)
NORM = colors.Normalize(vmin=0, vmax=9)


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
            # print(inputs_batch.shape) # torch.Size([64, 8])
            outputs = D(inputs_batch)
            # print(outputs.shape)    # torch.Size([32])
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


def get_features(nets, selected, args, net_dataidx_map, test_dl=None, device="cpu"):
    global features  # 声明为全局变量
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

                # 假设 features 是 net 的某一层的输出（你可能需要使用 hook 或其他方式来获取）
                # 将每个批次的 features 转移到 CPU 以节省 GPU 内存
                # features = out.detach().cpu()
                mid_output = features
                features = mid_output.detach().cpu()

                # 将 features 添加到列表中，而不是在 GPU 上拼接
                G_output_list.append(features)

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
            real_samples.append(features.cpu())

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


def local_train_net_fedtopo(nets, selected, args, net_dataidx_map, global_model, history, round, pi, test_dl=None, device="cpu"):
    avg_acc = 0.0

    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)
        global_model.to(device)

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

        # trainacc, testacc = train_net_fedtopo(net_id, net, train_dl_local, test_dl, n_epoch, args.lr,
        #                                        args.optimizer, global_model, history, round,
        #                                        device=device)
        # 轮数越大，alpha越大（可控上线）
        alpha = min(0.05 + round * 0.01, 0.5)  # 最高到0.5

        trainacc, testacc = train_net_fedtopo(
            net_id, net, train_dl_local, test_dl, n_epoch, args.lr, args.optimizer,
            global_model, history, round, device=device, pi=pi, K=2, pool_size=8, alpha=alpha)
        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        gc.collect()
        torch.cuda.empty_cache()
    avg_acc /= len(selected)
    if args.alg == 'local_training':
        logger.info("avg test acc %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list


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
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
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
    key_rounds = [0, args.comm_round // 4, args.comm_round // 2, (3 * args.comm_round - 2) // 4, args.comm_round - 1]

    if args.alg == 'base':

        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        # 初始化history，用于绘图
        history = {
            'client_total_loss': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_ce_loss': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_topo_loss': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_train_acc': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_test_acc': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_topo_distance': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_similarity': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_entropy': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_delta_topo_dist': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'round_var': [0.0] * (args.comm_round + 1),
            'client_cka': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_swd': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
        }

        # 设置需要保存PI的轮次
        pi_records = {}  # {round: {client_id: pi_vec, 'global': pi_vec}}

        # ---- 先对全局模型用全局训练集采样，对 PI fit 一次！----
        print("Fitting PersistenceImage on global_model + train_dl_global ...")
        pi = fit_persistence_image_from_loader(
            global_model, train_dl_global, device,
            layer_name=args.feature_layer, pool_size=8, K=2, max_batches=20  # 按你需求可调参数
        )
        print("Done fitting PersistenceImage.")

        for round in range(args.comm_round):
            print(f"[进程实际内存] {(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024):.1f} MB")
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

            local_train_net_fedtopo(nets, selected, args, net_dataidx_map, global_model, history, round, pi, nets_train_dl_local, test_dl=test_dl_global, device=device)

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
            global_model.eval()
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)

            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            # 计算全局模型在全局数据集上的熵
            if args.dataset != 'generated':
                entropy = compute_global_entropy(global_model, train_dl_global, device=device)
                logger.info('>> Global Model Entropy: %f' % entropy)

            # ------------------------ 可视化主流程 ------------------------
            if round in key_rounds:
                # 0️⃣ 统一提特征（一次前向）
                x, y = get_umap_sample_x(
                    train_dl_global,
                    n_batches=args.n_umap_batches,
                    per_batch=32,
                    device=args.device
                )

                model_names = ['global'] + [f'client{cid}' for cid in range(args.n_parties)]
                models_all = [global_model] + [nets[k] for k in nets]

                model_feats = {}  # {name: tensor(N,C,H,W)}
                with torch.no_grad():
                    for name, model in zip(model_names, models_all):
                        model = model.to(args.device)
                        model.eval()
                        feat = extract_layer_features(
                            model, x,
                            layer_name=args.feature_layer,
                            pool_size=8,
                            device=args.device
                        )
                        model_feats[name] = feat.cpu()  # 存到 CPU
                        torch.cuda.empty_cache()

                # ============== 1. 拓扑条形图 & PI =================
                global_feat = model_feats['global']
                client_feats = [model_feats[f'client{cid}'] for cid in range(args.n_parties)]

                vis_items = []  # [(name, bars, pi_vec)]
                g_arr = global_feat[0, 0].numpy()
                g_bars, g_pi = compute_barcode_and_pi(g_arr, pi)
                vis_items.append(("Global", g_bars, g_pi))

                for cid, feat in enumerate(client_feats):
                    c_arr = feat[0, 0].numpy()
                    c_bars, c_pi = compute_barcode_and_pi(c_arr, pi)
                    vis_items.append((f"Client{cid}", c_bars, c_pi))

                vmin, vmax = 0, max(p.max() for _, _, p in vis_items)  # 统一 PI 色阶
                FILTRATION_RANGE = get_filtration_range([b for _, b, _ in vis_items])

                fig, axs = plt.subplots(len(vis_items), 2,
                                        figsize=(8, 2.8 * len(vis_items)))

                legend_handles, legend_labels = [], []
                for idx, (lbl, bars, pi_vec) in enumerate(vis_items):
                    handles, labels_ = plot_barcode(
                        bars, axs[idx, 0],
                        title=f"{lbl} Barcode",
                        linewidth=5, xlim=FILTRATION_RANGE,
                        colors=custom_colors,
                        title_fontsize=16, label_fontsize=15,
                        legend_fontsize=14, show_legend=False
                    )
                if idx == 0:
                    legend_handles, legend_labels = handles, labels_

                    plot_pi(
                        pi_vec, axs[idx, 1],
                        title=f"{lbl} PI",
                        cmap=morandi_deepblue_cmap_r,
                        vmin=vmin, vmax=vmax,
                        title_fontsize=16, label_fontsize=15,
                        colorbar_fontsize=13
                    )
                    axs[idx, 0].set_ylabel(lbl, fontsize=15)

                if legend_handles:
                    # 给 legend 预留的高度
                    reserved = 0.04  # 8 % 画布高度

                    # 先让 tight_layout 只在 1‑reserved 之上排版
                    plt.tight_layout(rect=(0, reserved, 1, 1))

                    # 再把 legend 放到预留区的中间 (x=0.5, y=reserved/2)
                    fig.legend(
                        legend_handles, legend_labels,
                        loc='lower center',
                        bbox_to_anchor=(0.5, reserved / 2),
                        bbox_transform=fig.transFigure,  # 以 fig 坐标解释 bbox
                        ncol=len(legend_handles),
                        fontsize=14,
                        frameon=False
                    )

                plt.savefig(f"logs/topo_compare_round{round}.png",
                            dpi=400, bbox_inches="tight")
                plt.close(fig)
                print(f"[可视化] 已保存 logs/topo_compare_round{round}.png")

                # ================= 2. UMAP 可视化 ==================
                # 2-1 拼接特征
                all_features, all_labels = [], []
                for name in model_names:
                    feat_np = model_feats[name].view(model_feats[name].shape[0], -1).numpy()
                    all_features.append(feat_np)
                    all_labels += [name] * feat_np.shape[0]
                feats = np.vstack(all_features)
                labels = np.array(all_labels)

                # 2-2 处理 y，使长度匹配 feats
                if hasattr(y, 'cpu'):
                    y = y.cpu().numpy()
                elif isinstance(y, list):
                    y = np.array(y)
                y = np.tile(y, len(model_names))  # 复制到每个模型

                if feats.shape[0] > args.max_samples:  # 可选截断
                    feats = feats[:args.max_samples]
                    labels = labels[:args.max_samples]
                    y = y[:args.max_samples]

                # 2-3 降维
                X_emb2d = umap.UMAP(n_components=2, random_state=42).fit_transform(feats)
                X_emb3d = umap.UMAP(n_components=3, random_state=42).fit_transform(feats)

                # ------------- 2‑D 散点 -----------------
                colors_umap = ['#DB7093', '#9370DB', '#6495ED',
                               '#90EE90', '#F0E68C', '#F4A460']

                fig, ax = plt.subplots(figsize=(8, 7))

                # ① 先画所有非‑global 的点
                for idx, name in enumerate(model_names):
                    if name == 'global':
                        continue  # 留到最后
                    mask = labels == name
                    ax.scatter(
                        X_emb2d[mask, 0], X_emb2d[mask, 1],
                        c=colors_umap[idx % len(colors_umap)],
                        marker='o', s=60, alpha=0.5,
                        edgecolors='none', zorder=1,  # 较低 zorder
                        label=name.capitalize()
                    )

                # ② 再单独画 global，给很高的 zorder
                g_mask = labels == 'global'
                ax.scatter(
                    X_emb2d[g_mask, 0], X_emb2d[g_mask, 1],
                    c=colors_umap[0],
                    marker='x', s=70, alpha=0.7,
                    linewidths=2.5,
                    edgecolors='none', zorder=100,  # 最高 zorder
                    label='Global'
                )

                # ------- 布局与 legend -------
                rows = 2
                ncols = 3
                reserved = 0.15  # 预留 12 % 画布高度，留得更宽一些

                # ①：把子图都压到剩下 1‑reserved 的区域
                fig.subplots_adjust(bottom=reserved)  # 和 tight_layout(rect=…) 等价但更直观

                # ②：把 legend 放在预留区正中
                handles, labels_ = ax.get_legend_handles_labels()
                fig.legend(
                    handles, labels_,
                    loc='lower center',
                    bbox_to_anchor=(0.5, reserved / 5),  # 纵坐标仍在预留区正中
                    bbox_transform=fig.transFigure,
                    ncol=ncols, handlelength=1.5,
                    columnspacing=1.0, handletextpad=0.5,
                    frameon=False, fontsize=16
                )

                ax.set_title(f'UMAP 2D feature projection (Round {round})', fontsize=20)
                ax.set_xlabel('UMAP-1', fontsize=18)
                ax.set_ylabel('UMAP-2', fontsize=18)
                ax.tick_params(axis='both', labelsize=16)

                # 给底部预留空间容纳 legend
                fig.tight_layout(rect=[0, 0.08, 1, 1])

                fig.savefig(f"logs/umap2d_compare_round{round}.png", dpi=400)
                plt.close(fig)
                print(f"[UMAP 2D可视化] 已保存 logs/umap2d_compare_round{round}.png")

                # ------------- 3‑D 总图 ------------------
                fig = plt.figure(figsize=(8, 7))
                ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

                # ① 先画各 client
                for idx, name in enumerate(model_names):
                    if name == 'global':
                        continue
                    mask = labels == name
                    ax.scatter(
                        X_emb3d[mask, 0], X_emb3d[mask, 1], X_emb3d[mask, 2],
                        c=colors_umap[idx % len(colors_umap)],
                        s=40, alpha=0.5, edgecolors='none',
                        zorder=1, label=name.capitalize()
                    )

                # ② 再单独画 global
                g_mask = labels == 'global'
                ax.scatter(
                    X_emb3d[g_mask, 0], X_emb3d[g_mask, 1], X_emb3d[g_mask, 2],
                    c=colors_umap[0],  # 颜色照旧
                    marker='x', s=50, alpha=0.7,  # s 稍大更显眼
                    linewidths=2.5,  # 用线宽来控制粗细
                    zorder=100, label='Global'
                )

                # legend 现在就能包含 Global 了
                ax.legend(fontsize=16, loc='best', frameon=True)

                # —— 其余保持不变 ——
                ax.set_title(f'UMAP 3D feature projection (Round {round})',
                             fontsize=18, pad=12)
                ax.set_xlabel('UMAP-1', fontsize=16, labelpad=16)
                ax.set_ylabel('UMAP-2', fontsize=16, labelpad=12)
                ax.set_zlabel('UMAP-3', fontsize=16, labelpad=12)

                ax.locator_params(axis='x', nbins=15)
                ax.locator_params(axis='y', nbins=15)
                ax.locator_params(axis='z', nbins=15)
                ax.tick_params(axis='both', labelsize=14)
                ax.grid(True)

                # ―― 旋转 UMAP‑1（x 轴）刻度标签 ――
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(45)  # 45° 斜排
                    lbl.set_horizontalalignment('right')

                plt.tight_layout()
                plt.savefig(f"logs/umap3d_compare_round{round}.png", dpi=400)
                plt.close(fig)
                print(f"[UMAP 3D可视化] 已保存 logs/umap3d_compare_round{round}.png")

                # ------------- 3-D 各模型子图 -------------
                color_per_class = [
                    '#CD5C5C', '#F4A460', '#F0E68C', '#90EE90', '#6495ED',
                    '#9370DB', '#808080', '#FFB6C1', '#48D1CC', '#BDB76B'
                ]
                cifar10_labels = [
                    "airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"
                ]

                fig = plt.figure(figsize=(8 * len(model_names), 8))
                axs = [fig.add_subplot(1, len(model_names), i + 1, projection='3d')
                       for i in range(len(model_names))]

                for ax, name in zip(axs, model_names):
                    mask = (labels == name)
                    X_model = X_emb3d[mask]
                    y_model = y[mask]

                    for cls in np.unique(y_model):
                        c_mask = (y_model == cls)
                        ax.scatter(
                            X_model[c_mask, 0], X_model[c_mask, 1], X_model[c_mask, 2],
                            c=color_per_class[int(cls) % len(color_per_class)],
                            s=60, label=cifar10_labels[int(cls)],
                            alpha=0.78, edgecolors='none'
                        )

                    # —— 调整标题与坐标轴标题的位置 ——
                    ax.set_title(f"{name} UMAP 3D (Round {round})",
                                 fontsize=18, pad=0)  # pad ↓ 让标题稍微下移
                    ax.set_xlabel('UMAP-1', fontsize=16, labelpad=12)  # labelpad ↑
                    ax.set_ylabel('UMAP-2', fontsize=16, labelpad=12)
                    ax.set_zlabel('UMAP-3', fontsize=16, labelpad=12)

                    ax.locator_params(axis='x', nbins=8)
                    ax.locator_params(axis='y', nbins=8)
                    ax.locator_params(axis='z', nbins=8)
                    ax.tick_params(axis='both', labelsize=14)

                # —— 合并图例 ——
                handles, labels_ = axs[0].get_legend_handles_labels()
                by_label = dict(zip(labels_, handles))
                fig.legend(
                    list(by_label.values()), list(by_label.keys()),
                    fontsize=16, loc='center left',
                    bbox_to_anchor=(0.92, 0.5), borderaxespad=0.
                )
                fig.tight_layout(rect=(0, 0, 0.90, 1), w_pad=2.5)
                plt.savefig(f"logs/umap3d_all_round{round}.png", dpi=400, bbox_inches="tight")
                plt.close()
                print(f"[3D-UMAP可视化] 已保存 logs/umap3d_all_round{round}.png")

                # ----------- 释放显存 -----------
                torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        # 最后再画一下 loss 曲线
        # TODO: 去掉最后的10
        fig3 = plot_training_progress(history, args.n_parties, args.comm_round)
        fig3.savefig(os.path.join(args.logdir, 'training_progress.png'))
        plt.close(fig3)  # 关闭图形
        # 节约内存这一块
        plt.close('all')
        gc.collect()

    elif args.alg == 'fedtopo':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        # 初始化history，用于绘图
        history = {
            'client_total_loss': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_ce_loss': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_topo_loss': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_train_acc': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_test_acc': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_topo_distance': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_similarity': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_entropy': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_delta_topo_dist': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'round_var': [0.0] * (args.comm_round + 1),
            'client_cka': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_swd': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
        }

        MAX_SAMPLES = 1000 # 定义最大样本数
        rng = np.random.RandomState(42)  # 定义随机数生成器

        # 初始化 client_features_list 和 client_targets_list
        client_features_list = []
        client_targets_list = []

        # 生成初始全局特征
        global_model.eval()
        global_features, global_targets = collect_global_features(global_model, train_dl_global, device)
        if len(global_features) > MAX_SAMPLES:
            idx = rng.choice(len(global_features), MAX_SAMPLES, replace=False)
            global_features = global_features[idx]
            global_targets = global_targets[idx]  # 同步下采样标签
        # ---- 在 federated_learning 最开始，做一次全局 fit ----
        umap_reducer = umap.UMAP(n_components=3, random_state=42).fit(global_features.cpu().numpy())

        # 设置需要保存PI的轮次
        pi_records = {}  # {round: {client_id: pi_vec, 'global': pi_vec}}

        # ---- 先对全局模型用全局训练集采样，对 PI fit 一次！----
        print("Fitting PersistenceImage on global_model + train_dl_global ...")
        pi = fit_persistence_image_from_loader(
            global_model, train_dl_global, device,
            layer_name='layer3', pool_size=8, K=2, max_batches=20  # 按你需求可调参数
        )
        print("Done fitting PersistenceImage.")

        for round in range(args.comm_round):
            print(f"[进程实际内存] {(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024):.1f} MB")
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

            local_train_net_fedtopo(nets, selected, args, net_dataidx_map, global_model, history, round, pi, test_dl=test_dl_global, device=device)

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
            global_model.eval()
            train_acc = compute_accuracy(global_model, train_dl_global, device=device)
            test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True,
                                                     device=device)
            logger.info('>> Global Model Train accuracy: %f' % train_acc)
            logger.info('>> Global Model Test accuracy: %f' % test_acc)

            # 计算全局模型在全局数据集上的熵
            if args.dataset != 'generated':
                entropy = compute_global_entropy(global_model, train_dl_global, device=device)
                logger.info('>> Global Model Entropy: %f' % entropy)

            # ------------------------ 可视化主流程 ------------------------
            if round in key_rounds:
                # 0️⃣ 统一提特征（一次前向）
                x, y = get_umap_sample_x(
                    train_dl_global,
                    n_batches=args.n_umap_batches,
                    per_batch=32,
                    device=args.device
                )

                model_names = ['global'] + [f'client{cid}' for cid in range(args.n_parties)]
                models_all = [global_model] + [nets[k] for k in nets]

                model_feats = {}  # {name: tensor(N,C,H,W)}
                with torch.no_grad():
                    for name, model in zip(model_names, models_all):
                        model = model.to(args.device)
                        model.eval()
                        feat = extract_layer_features(
                            model, x,
                            layer_name=args.feature_layer,
                            pool_size=8,
                            device=args.device
                        )
                        model_feats[name] = feat.cpu()  # 存到 CPU
                torch.cuda.empty_cache()

                # ============== 1. 拓扑条形图 & PI =================
                global_feat = model_feats['global']
                client_feats = [model_feats[f'client{cid}'] for cid in range(args.n_parties)]

                vis_items = []  # [(name, bars, pi_vec)]
                g_arr = global_feat[0, 0].numpy()
                g_bars, g_pi = compute_barcode_and_pi(g_arr, pi)
                vis_items.append(("Global", g_bars, g_pi))

                for cid, feat in enumerate(client_feats):
                    c_arr = feat[0, 0].numpy()
                    c_bars, c_pi = compute_barcode_and_pi(c_arr, pi)
                    vis_items.append((f"Client{cid}", c_bars, c_pi))

                vmin, vmax = 0, max(p.max() for _, _, p in vis_items)  # 统一 PI 色阶
                FILTRATION_RANGE = get_filtration_range([b for _, b, _ in vis_items])

                fig, axs = plt.subplots(len(vis_items), 2,
                                        figsize=(8, 2.8 * len(vis_items)))

                legend_handles, legend_labels = [], []
                for idx, (lbl, bars, pi_vec) in enumerate(vis_items):
                    handles, labels_ = plot_barcode(
                        bars, axs[idx, 0],
                        title=f"{lbl} Barcode",
                        linewidth=5, xlim=FILTRATION_RANGE,
                        colors=custom_colors,
                        title_fontsize=16, label_fontsize=15,
                        legend_fontsize=14, show_legend=False
                    )
                    if idx == 0:
                        legend_handles, legend_labels = handles, labels_

                    plot_pi(
                        pi_vec, axs[idx, 1],
                        title=f"{lbl} PI",
                        cmap=morandi_deepblue_cmap_r,
                        vmin=vmin, vmax=vmax,
                        title_fontsize=16, label_fontsize=15,
                        colorbar_fontsize=13
                    )
                    axs[idx, 0].set_ylabel(lbl, fontsize=15)

                if legend_handles:
                    # 给 legend 预留的高度
                    reserved = 0.04  # 8 % 画布高度

                    # 先让 tight_layout 只在 1‑reserved 之上排版
                    plt.tight_layout(rect=(0, reserved, 1, 1))

                    # 再把 legend 放到预留区的中间 (x=0.5, y=reserved/2)
                    fig.legend(
                        legend_handles, legend_labels,
                        loc='lower center',
                        bbox_to_anchor=(0.5, reserved / 2),
                        bbox_transform=fig.transFigure,  # 以 fig 坐标解释 bbox
                        ncol=len(legend_handles),
                        fontsize=14,
                        frameon=False
                    )

                plt.savefig(f"logs/topo_compare_round{round}.png",
                            dpi=400, bbox_inches="tight")
                plt.close(fig)
                print(f"[可视化] 已保存 logs/topo_compare_round{round}.png")

                # ================= 2. UMAP 可视化 ==================
                # 2-1 拼接特征
                all_features, all_labels = [], []
                for name in model_names:
                    feat_np = model_feats[name].view(model_feats[name].shape[0], -1).numpy()
                    all_features.append(feat_np)
                    all_labels += [name] * feat_np.shape[0]
                feats = np.vstack(all_features)
                labels = np.array(all_labels)

                # 2-2 处理 y，使长度匹配 feats
                if hasattr(y, 'cpu'):
                    y = y.cpu().numpy()
                elif isinstance(y, list):
                    y = np.array(y)
                y = np.tile(y, len(model_names))  # 复制到每个模型

                if feats.shape[0] > args.max_samples:  # 可选截断
                    feats = feats[:args.max_samples]
                    labels = labels[:args.max_samples]
                    y = y[:args.max_samples]

                # 2-3 降维
                X_emb2d = umap.UMAP(n_components=2, random_state=42).fit_transform(feats)
                X_emb3d = umap.UMAP(n_components=3, random_state=42).fit_transform(feats)

                # ------------- 2‑D 散点 -----------------
                colors_umap = ['#DB7093', '#9370DB', '#6495ED',
                               '#90EE90', '#F0E68C', '#F4A460']

                fig, ax = plt.subplots(figsize=(8, 7))

                # ① 先画所有非‑global 的点
                for idx, name in enumerate(model_names):
                    if name == 'global':
                        continue  # 留到最后
                    mask = labels == name
                    ax.scatter(
                        X_emb2d[mask, 0], X_emb2d[mask, 1],
                        c=colors_umap[idx % len(colors_umap)],
                        marker='o', s=60, alpha=0.5,
                        edgecolors='none', zorder=1,  # 较低 zorder
                        label=name.capitalize()
                    )

                # ② 再单独画 global，给很高的 zorder
                g_mask = labels == 'global'
                ax.scatter(
                    X_emb2d[g_mask, 0], X_emb2d[g_mask, 1],
                    c=colors_umap[0],
                    marker='x', s=70, alpha=0.7,
                    linewidths=2.5,
                    edgecolors='none', zorder=100,  # 最高 zorder
                    label='Global'
                    )

                # ------- 布局与 legend -------
                rows = 2
                ncols = 3
                reserved = 0.15  # 预留 12 % 画布高度，留得更宽一些

                # ①：把子图都压到剩下 1‑reserved 的区域
                fig.subplots_adjust(bottom=reserved)  # 和 tight_layout(rect=…) 等价但更直观

                # ②：把 legend 放在预留区正中
                handles, labels_ = ax.get_legend_handles_labels()
                fig.legend(
                    handles, labels_,
                    loc='lower center',
                    bbox_to_anchor=(0.5, reserved / 5),  # 纵坐标仍在预留区正中
                    bbox_transform=fig.transFigure,
                    ncol=ncols, handlelength=1.5,
                    columnspacing=1.0, handletextpad=0.5,
                    frameon=False, fontsize=16
                )

                ax.set_title(f'UMAP 2D feature projection (Round {round})', fontsize=20)
                ax.set_xlabel('UMAP-1', fontsize=18)
                ax.set_ylabel('UMAP-2', fontsize=18)
                ax.tick_params(axis='both', labelsize=16)

                fig.savefig(f"logs/umap2d_compare_round{round}.png", dpi=400)
                plt.close(fig)
                print(f"[UMAP 2D可视化] 已保存 logs/umap2d_compare_round{round}.png")

                # ------------- 3‑D 总图 ------------------
                fig = plt.figure(figsize=(8, 7))
                ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

                # ① 先画各 client
                for idx, name in enumerate(model_names):
                    if name == 'global':
                        continue
                    mask = labels == name
                    ax.scatter(
                        X_emb3d[mask, 0], X_emb3d[mask, 1], X_emb3d[mask, 2],
                        c=colors_umap[idx % len(colors_umap)],
                        s=40, alpha=0.5, edgecolors='none',
                        zorder=1, label=name.capitalize()
                    )

                # ② 再单独画 global
                g_mask = labels == 'global'
                ax.scatter(
                    X_emb3d[g_mask, 0], X_emb3d[g_mask, 1], X_emb3d[g_mask, 2],
                    c=colors_umap[0],  # 颜色照旧
                    marker='x', s=50, alpha=0.7,  # s 稍大更显眼
                    linewidths=2.5,  # 用线宽来控制粗细
                    zorder=100, label='Global'
                )

                # legend 现在就能包含 Global 了
                ax.legend(fontsize=16, loc='best', frameon=True)

                # —— 其余保持不变 ——
                ax.set_title(f'UMAP 3D feature projection (Round {round})',
                             fontsize=18, pad=12)
                ax.set_xlabel('UMAP-1', fontsize=16, labelpad=16)
                ax.set_ylabel('UMAP-2', fontsize=16, labelpad=12)
                ax.set_zlabel('UMAP-3', fontsize=16, labelpad=12)

                ax.locator_params(axis='x', nbins=15)
                ax.locator_params(axis='y', nbins=15)
                ax.locator_params(axis='z', nbins=15)

                ax.tick_params(axis='both', labelsize=14)
                ax.grid(True)

                # ―― 旋转 UMAP‑1（x 轴）刻度标签 ――
                for lbl in ax.get_xticklabels():
                    lbl.set_rotation(45)  # 45° 斜排
                    lbl.set_horizontalalignment('right')

                plt.tight_layout()
                plt.savefig(f"logs/umap3d_compare_round{round}.png", dpi=400)
                plt.close(fig)
                print(f"[UMAP 3D可视化] 已保存 logs/umap3d_compare_round{round}.png")

                # ------------- 3-D 各模型子图 -------------
                color_per_class = [
                    '#CD5C5C', '#F4A460', '#F0E68C', '#90EE90', '#6495ED',
                    '#9370DB', '#808080', '#FFB6C1', '#48D1CC', '#BDB76B'
                ]
                cifar10_labels = [
                    "airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"
                ]

                fig = plt.figure(figsize=(8 * len(model_names), 8))
                axs = [fig.add_subplot(1, len(model_names), i + 1, projection='3d')
                       for i in range(len(model_names))]

                for ax, name in zip(axs, model_names):
                    mask = (labels == name)
                    X_model = X_emb3d[mask]
                    y_model = y[mask]

                    for cls in np.unique(y_model):
                        c_mask = (y_model == cls)
                        ax.scatter(
                            X_model[c_mask, 0], X_model[c_mask, 1], X_model[c_mask, 2],
                            c=color_per_class[int(cls) % len(color_per_class)],
                            s=60, label=cifar10_labels[int(cls)],
                            alpha=0.78, edgecolors='none'
                        )

                    # —— 调整标题与坐标轴标题的位置 ——
                    ax.set_title(f"{name} UMAP 3D (Round {round})",
                                 fontsize=18, pad=0)  # pad ↓ 让标题稍微下移
                    ax.set_xlabel('UMAP-1', fontsize=16, labelpad=12)  # labelpad ↑
                    ax.set_ylabel('UMAP-2', fontsize=16, labelpad=12)
                    ax.set_zlabel('UMAP-3', fontsize=16, labelpad=12)

                    ax.locator_params(axis='x', nbins=8)
                    ax.locator_params(axis='y', nbins=8)
                    ax.locator_params(axis='z', nbins=8)
                    ax.tick_params(axis='both', labelsize=14)

                # —— 合并图例 ——
                handles, labels_ = axs[0].get_legend_handles_labels()
                by_label = dict(zip(labels_, handles))
                fig.legend(
                    list(by_label.values()), list(by_label.keys()),
                    fontsize=16, loc='center left',
                    bbox_to_anchor=(0.92, 0.5), borderaxespad=0.
                )
                fig.tight_layout(rect=(0, 0, 0.90, 1), w_pad=2.5)
                plt.savefig(f"logs/umap3d_all_round{round}.png", dpi=400, bbox_inches="tight")
                plt.close()
                print(f"[3D-UMAP可视化] 已保存 logs/umap3d_all_round{round}.png")

                # ----------- 释放显存 -----------
                torch.cuda.empty_cache()

            gc.collect()
            torch.cuda.empty_cache()

        # 最后再画一下 loss 曲线
        # TODO: 去掉最后的10
        fig3 = plot_training_progress(history, args.n_parties, args.comm_round)
        fig3.savefig(os.path.join(args.logdir, 'training_progress.png'))
        plt.close(fig3)  # 关闭图形
        # 节约内存这一块
        plt.close('all')
        gc.collect()

    elif args.alg == 'fedavg':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_parties, args)
        global_models, global_model_meta_data, global_layer_type = init_nets(args.net_config, 0, 1, args)
        global_model = global_models[0]

        global_para = global_model.state_dict()

        if args.is_same_initial:
            for net_id, net in nets.items():
                net.load_state_dict(global_para)

        # 设置需要保存PI的轮次
        pi_records = {}  # {round: {client_id: pi_vec, 'global': pi_vec}}

        # ---- 先对全局模型用全局训练集采样，对 PI fit 一次！----
        print("Fitting PersistenceImage on global_model + train_dl_global ...")
        pi = fit_persistence_image_from_loader(
            global_model, train_dl_global, device,
            layer_name=args.feature_layer, pool_size=8, K=2, max_batches=20  # 按你需求可调参数
        )
        print("Done fitting PersistenceImage.")

        # 初始化history，用于绘图
        history = {
            'client_total_loss': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_ce_loss': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_topo_loss': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_train_acc': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_test_acc': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_topo_distance': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_similarity': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_entropy': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_delta_topo_dist': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'round_var': [0.0] * (args.comm_round + 1),
            'client_cka': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
            'client_swd': [[0.0] * (args.comm_round + 1) for _ in range(args.n_parties)],
        }

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

            # 计算全局模型在全局数据集上的熵
            if args.dataset != 'generated':
                entropy = compute_global_entropy(global_model, train_dl_global, device=device)
                logger.info('>> Global Model Entropy: %f' % entropy)

            pi_records = {}
            if round in key_rounds:
                # --- 每个客户端 ---
                x, _ = next(iter(train_dl_global))
                x = x[:32].to(args.device)  # 可设batch大小
                with torch.no_grad():
                    client_feats = []
                    for net_id, net in nets.items():
                        net = net.to(args.device)  # 把模型放到同一个 device 上
                        net.eval()
                        feat = extract_layer_features(net, x, layer_name=args.feature_layer, pool_size=8,
                                                      device=args.device)
                        client_feats.append(feat.cpu())  # 先转到CPU，节省显存
                        # 清理显存
                        torch.cuda.empty_cache()

                    # --- 全局模型 ---
                    global_model = global_model.to(args.device)  # 保证全局模型也在对的设备
                    global_model.eval()
                    global_feat = extract_layer_features(global_model, x, layer_name=args.feature_layer, pool_size=8,
                                                         device=args.device)

                    # 清理全局特征数据
                    torch.cuda.empty_cache()

                # ----开始可视化-----
                # 全局模型特征转CPU
                global_feat = global_feat.cpu()

                # 1. 一次性先处理并存储所有 bars/pi，避免重复计算
                vis_items = []  # [(name, bars, pi)]
                # 全局
                g_arr = global_feat[0, 0].numpy()
                g_bars, g_pi = compute_barcode_and_pi(g_arr, pi)
                vis_items.append(("Global", g_bars, g_pi))

                # 各client
                for cid, feat in enumerate(client_feats):
                    c_arr = feat[0, 0].numpy()
                    c_bars, c_pi = compute_barcode_and_pi(c_arr, pi)
                    vis_items.append((f"Client{cid}", c_bars, c_pi))

                # 2. 统一统计横轴范围（只遍历一遍）
                all_bars = [item[1] for item in vis_items]
                FILTRATION_RANGE = get_filtration_range(all_bars)  # 上面提供的自动统计函数

                # 3. 开始画图（也只遍历一遍）
                fig, axs = plt.subplots(len(vis_items), 2, figsize=(10, 3.5 * len(vis_items)))
                for idx, (label, bars, pi_vec) in enumerate(vis_items):
                    plot_barcode(bars, axs[idx, 0], title=f"{label} Barcode", linewidth=5,
                                 xlim=FILTRATION_RANGE, colors=custom_colors)
                    plot_pi(pi_vec, axs[idx, 1], title=f"{label} PI", cmap=morandi_blue_cmap)
                    axs[idx, 0].set_ylabel(label, fontsize=12)

                plt.tight_layout()
                plt.savefig(f"logs/topo_compare_round{round}.png", dpi=300)
                plt.close(fig)
                print(f"[可视化] 已保存 logs/topo_compare_round{round}.png")

                # 清理显存
                torch.cuda.empty_cache()

            gc.collect()
            torch.cuda.empty_cache()

        # 最后再画一下 loss 曲线
        # TODO: 去掉最后的10
        fig3 = plot_training_progress(history, args.n_parties, args.comm_round)
        fig3.savefig(os.path.join(args.logdir, 'training_progress.png'))
        plt.close(fig3)  # 关闭图形

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
