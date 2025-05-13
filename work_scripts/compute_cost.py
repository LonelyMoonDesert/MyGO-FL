import torch
import torch.nn as nn
from thop import profile
from torchsummary import summary
from vggmodel import vgg11
from resnetcifar import ResNet18_cifar10
from model import DiscriminatorS, Discriminator_vgg11_cifar10

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化客户端模型
client_model_resnet = ResNet18_cifar10(num_classes=10).to(device)
client_model_vgg = vgg11().to(device)

# 初始化服务端判别器
discriminator_resnet = DiscriminatorS(input_channels=64).to(device)
discriminator_vgg = Discriminator_vgg11_cifar10(input_channels=512).to(device)

# 定义输入张量大小
input_size = (3, 32, 32)  # CIFAR10 输入大小
input_tensor = torch.randn(1, *input_size).to(device)

# 定义特征生成器
# ResNet18 特征生成器：到 layer3
resnet_feature_extractor = torch.nn.Sequential(
    client_model_resnet.conv1,
    client_model_resnet.bn1,
    client_model_resnet.relu,
    client_model_resnet.layer1,
    client_model_resnet.layer2,
    client_model_resnet.layer3
).to(device)

# VGG11 特征生成器：到 features[20]
vgg_feature_extractor = torch.nn.Sequential(
    *list(client_model_vgg.features[:21])
).to(device)

# 计算 ResNet18 特征生成器的参数量和 MACs
print("\nFeature Generator for ResNet18")
summary(resnet_feature_extractor, (3, 32, 32))
resnet_feature_macs, resnet_feature_params = profile(resnet_feature_extractor, inputs=(input_tensor,))
print(f"ResNet18 Feature Generator - Parameters: {resnet_feature_params / 1e6:.2f}M, MACs: {resnet_feature_macs / 1e9:.2f}G")

# 计算 VGG11 特征生成器的参数量和 MACs
print("\nFeature Generator for VGG11")
summary(vgg_feature_extractor, (3, 32, 32))
vgg_feature_macs, vgg_feature_params = profile(vgg_feature_extractor, inputs=(input_tensor,))
print(f"VGG11 Feature Generator - Parameters: {vgg_feature_params / 1e6:.2f}M, MACs: {vgg_feature_macs / 1e9:.2f}G")

# 计算 ResNet18 的参数量和 MACs
print("\nClient Model: ResNet18")
summary(client_model_resnet, input_size)
resnet_macs, resnet_params = profile(client_model_resnet, inputs=(input_tensor,))
print(f"ResNet18 - Parameters: {resnet_params / 1e6:.2f}M, MACs: {resnet_macs / 1e9:.2f}G")

# 计算 VGG11 的参数量和 MACs
print("\nClient Model: VGG11")
summary(client_model_vgg, input_size)
vgg_macs, vgg_params = profile(client_model_vgg, inputs=(input_tensor,))
print(f"VGG11 - Parameters: {vgg_params / 1e6:.2f}M, MACs: {vgg_macs / 1e9:.2f}G")

# 判别器输入张量大小
discriminator_input_resnet = torch.randn(1, 64, 4, 4).to(device)
discriminator_input_vgg = torch.randn(1, 512, 1, 1).to(device)

# 计算判别器（ResNet18 特征）
print("\nDiscriminator for ResNet18 Features")
summary(discriminator_resnet, (64, 4, 4))
discriminator_resnet_macs, discriminator_resnet_params = profile(discriminator_resnet, inputs=(discriminator_input_resnet,))
print(f"Discriminator for ResNet18 Features - Parameters: {discriminator_resnet_params / 1e6:.2f}M, MACs: {discriminator_resnet_macs / 1e9:.2f}G")

# 计算判别器（VGG11 特征）
print("\nDiscriminator for VGG11 Features")
summary(discriminator_vgg, (512, 1, 1))
discriminator_vgg_macs, discriminator_vgg_params = profile(discriminator_vgg, inputs=(discriminator_input_vgg,))
print(f"Discriminator for VGG11 Features - Parameters: {discriminator_vgg_params / 1e6:.2f}M, MACs: {discriminator_vgg_macs / 1e9:.2f}G")
