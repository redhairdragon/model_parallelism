from Model import ResNetBase
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
import torch
import torch.nn as nn
from torch.distributed.rpc import RRef
import torch.distributed.rpc as rpc
num_classes = 100


class ResNetShard1(ResNetBase):
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard1, self).__init__(
            Bottleneck, 64, num_classes=100, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7,
                      stride=2, padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 3),
            self._make_layer(128, 4, stride=2)
        ).to(self.device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.seq(x)
        return out.cpu()


class ResNetShard2(ResNetBase):
    def __init__(self, device, *args, **kwargs):
        super(ResNetShard2, self).__init__(
            Bottleneck, 512, num_classes=100, *args, **kwargs)

        self.device = device
        self.seq = nn.Sequential(
            self._make_layer(256, 6, stride=2),
            self._make_layer(512, 3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        ).to(self.device)

        self.fc = nn.Linear(512 * self._block.expansion,
                            num_classes).to(self.device)

    def forward(self, x_rref):
        x = x_rref.to_here().to(self.device)
        with self._lock:
            out = self.fc(torch.flatten(self.seq(x), 1))
        return out.cpu()


class DistResNet50(nn.Module):
    def __init__(self, num_split, workers, *args, **kwargs):
        super(DistResNet50, self).__init__()

        self.num_split = num_split

        # Put the first part of the ResNet50 on workers[0]
        self.p1_rref = rpc.remote(
            workers[0],
            ResNetShard1,
            args=("cuda:0",) + args,
            kwargs=kwargs
        )

        # Put the second part of the ResNet50 on workers[1]
        self.p2_rref = rpc.remote(
            workers[1],
            ResNetShard2,
            args=("cuda:0",) + args,
            kwargs=kwargs
        )

    def forward(self, xs):
        out_futures = []
        for x in iter(xs.split(self.num_split, dim=0)):
            x_rref = RRef(x)
            y_rref = self.p1_rref.remote().forward(x_rref)
            z_fut = self.p2_rref.rpc_async().forward(y_rref)
            out_futures.append(z_fut)

        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.p1_rref.remote().parameter_rrefs().to_here())
        remote_params.extend(self.p2_rref.remote().parameter_rrefs().to_here())
        return remote_params
