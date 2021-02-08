
import argparse
from deepspeed.pipe import PipelineModule
import torch
import torch.nn as nn
from torchvision.models import AlexNet
import deepspeed
import argparse
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import os

steps = 20


def add_argument():

    parser = argparse.ArgumentParser(description='CIFAR')

    # data
    # cuda
    parser.add_argument('--with_cuda',
                        default=False,
                        action='store_true',
                        help='use CPU in case there\'s no GPU support')
    parser.add_argument('--use_ema',
                        default=False,
                        action='store_true',
                        help='whether use exponential moving average')

    # train
    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args


def join_layers(vision_model):
    layers = [
        *vision_model.features,
        vision_model.avgpool,
        lambda x: torch.flatten(x, 1),
        *vision_model.classifier,
    ]
    return layers


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform)


deepspeed.init_distributed()
net = AlexNet(num_classes=10)
net = PipelineModule(layers=join_layers(net),
                     loss_fn=torch.nn.CrossEntropyLoss(),
                     num_stages=2,
                     partition_method="parameters",
                     activation_checkpoint_interval=0)


args = add_argument()
engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args, model=net, model_parameters=[p for p in net.parameters() if p.requires_grad], training_data=trainset)

for step in range(steps):
    loss = engine.train_batch()
    print(loss)
# deepspeed --hostfile=./hostfile model_parallel/deepspeed/tutorial.py --deepspeed --deepspeed_config model_parallel/deepspeed/ds_config.json
