import os
import torch
import sys
# import torch.distributed as dist
import torch.utils.data.distributed
from torchvision import datasets, transforms
import torch.nn.functional as F

# import torch.multiprocessing as mp
import time
import torch.distributed.autograd as dist_autograd
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from ModelShards import DistResNet50

import torch.distributed.rpc as rpc
import torch.nn as nn

batch_size = 32
num_classes = 100
lr = 0.01


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def run_master(num_split):
    print("run master")
    # put the two model parts on worker1 and worker2 respectively
    model = DistResNet50(
        num_split, ["worker1", "worker2"])
    loss_fn = nn.MSELoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
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
        val_dataset, batch_size=batch_size)

    batch_offset = 0
    for idx, (data, target) in enumerate(train_loader):
        print(f"Processing batch {idx}")
        batch_idx = batch_offset + idx
        # data, target = data.cuda(), target.cuda()
        target = target.view(-1, 1)
        labels = torch.zeros(batch_size, num_classes).scatter_(1, target, 1)
        with dist_autograd.context() as context_id:
            outputs = model(data)
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            opt.step(context_id)

        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                print("val loss:")
                print(F.cross_entropy(output, target))
                print("val accuracy:")
                print(accuracy(output, target))


def run_worker(rank, world_size, num_split):
    os.environ['MASTER_ADDR'] = '172.31.4.185'
    # os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=128)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(num_split)

    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        print("slave init")
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 3
    # dist.init_process_group(backend='gloo', init_method="tcp://172.31.56.27:23456",
    #                         rank=int(sys.argv[1]), world_size=world_size)

    num_split = 2
    tik = time.time()
    # mp.spawn(run_worker, args=(world_size, num_split),
    #  nprocs=world_size, join=True)
    run_worker(int(sys.argv[1]), world_size, num_split)
    # dist.barrier()
    tok = time.time()
    print(
        f"number of splits = {num_split}, execution time = {tok - tik}")
    time.sleep(3)
