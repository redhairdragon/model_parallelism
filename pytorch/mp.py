import os
import torch
import sys
# import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import datasets, transforms
import torch.nn.functional as F
from Metric import Metric

# import torch.multiprocessing as mp
import time
import torch.distributed.autograd as dist_autograd
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from ModelShards import DistResNet50

import torch.distributed.rpc as rpc
import torch.nn as nn
from tqdm import tqdm
import math
batch_size = 32
num_classes = 100
lr = 0.01
epoch = 40


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def run_master(num_split):
    print("run master")
    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet50(
        num_split, ["worker1", "worker2"])
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=lr, momentum=0.9
    )

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR100(root='./data', train=True,
                                      download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size)
    val_dataset = datasets.CIFAR100(root='./data', train=False,
                                    download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128)

    for e in range(epoch):
        model.train()
        train_loss = Metric("train_loss")
        train_accuracy = Metric("train_accuracy")
        with tqdm(
            total=len(train_loader),
            desc="Train Epoch #{}".format(e + 1),
        ) as t:
            for idx, (data, target) in enumerate(train_loader):
                with dist_autograd.context() as context_id:
                    outputs = model(data)
                    loss = F.cross_entropy(outputs, target)
                    dist_autograd.backward(context_id, [loss])
                    opt.step(context_id)
                    train_loss.update(loss)
                    train_accuracy.update(accuracy(outputs, target))
                    t.set_postfix(
                        {
                            "loss": train_loss.avg.item(),
                            "accuracy": 100.0 * train_accuracy.avg.item(),
                        }
                    )
                    t.update(1)

        model.eval()
        with tqdm(
            total=len(val_loader),
            desc="Valid Epoch #{}".format(e + 1),
        ) as t:
            with torch.no_grad():
                val_loss = Metric("val_loss")
                val_accuracy = Metric("val_accuracy")
                for data, target in val_loader:
                    output = model(data)
                    val_loss.update(F.cross_entropy(output, target))
                    val_accuracy.update(accuracy(output, target))
                    t.set_postfix(
                        {
                            "loss": val_loss.avg.item(),
                            "accuracy": 100.0 * val_accuracy.avg.item(),
                        }
                    )
                    t.update(1)


def run_worker(rank, world_size, num_split):
    os.environ['MASTER_ADDR'] = '172.31.13.136'
    # os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    # options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=32)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            # rpc_backend_options=options
        )
        run_master(num_split)

    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            # rpc_backend_options=options
        )
        print("slave init")
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 3
    print("starting")
    num_split = 2
    tik = time.time()
    run_worker(int(sys.argv[1]), world_size, num_split)
    tok = time.time()
