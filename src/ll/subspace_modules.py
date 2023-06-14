from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
from torch import Tensor


class SubspaceModule(nn.Module):
    members: nn.ModuleList

    def __init__(self, n, reinit):
        super().__init__()
        self.n = n
        self.reinit = reinit

    def _clone(self, m: nn.Module) -> nn.Module:
        return copy.deepcopy(m)

    def get_weight(self):
        raise NotImplementedError

    def _create_new_(self, module: nn.Module):
        module = self._clone(module)
        if self.reinit:
            module.reset_parameters()
        return module

    def reset_parameters(self):
        for m in self.members:
            m.reset_parameters()


class SubspaceConv(SubspaceModule):
    def __init__(self, module: nn.Conv2d, n: int, reinit: bool):
        super().__init__(n, reinit)
        self.members = nn.ModuleList([self._create_new_(module) for _ in range(n)])

        # copy the attributes of the conv layer
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.stride = module.stride
        self.has_bias = module.bias is not None

    def get_weight(self):
        w = sum([self.members[i].weight * self.alpha[i] for i in range(self.n)])
        if self.has_bias:
            b = sum([self.members[i].bias * self.alpha[i] for i in range(self.n)])
        else:
            b = None

        return w, b

    def __repr__(self):
        return "SubspaceConv(n={}, {}, {}, kernel_size={}, stride={})".format(
            self.n, self.in_channels, self.out_channels, self.kernel_size, self.stride
        )

    def retrieve_member(self, index):
        w = self.members[index].weight
        b = self.members[index].bias

        return w, b

    def retrieve_member_weight(self, index):
        return self.members[index].weight

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()
        x = F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
        return x


class SubspaceLinear(SubspaceModule):
    def __init__(self, module: nn.Linear, n: int, reinit: bool):
        super().__init__(n, reinit)
        self.members = nn.ModuleList([self._create_new_(module) for _ in range(n)])
        # self.bias = nn.ParameterList([self._create_new_(module) for _ in range(n)])

        # copy the attributes of the conv layer
        self.in_features = module.in_features
        self.out_features = module.out_features
        # self.alpha = [1 / n] * n

    def get_weight(self):
        w = sum([self.members[i].weight * self.alpha[i] for i in range(self.n)])
        b = sum([self.members[i].bias * self.alpha[i] for i in range(self.n)])

        return w, b

    def __repr__(self) -> str:
        return "SubspaceLinear(n={}, in_features={}, out_features={}, bias={})".format(
            self.n, self.in_features, self.out_features, self.members[0].bias is not None
        )

    def retrieve_member(self, index):
        w = self.members[index].weight
        b = self.members[index].bias

        return w, b

    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()
        x = F.linear(x, w, b)
        return x


class SubspaceBatchNorm2d(SubspaceModule):
    "pretty much copy pasted from official pytorch source code"

    def __init__(self, module: nn.BatchNorm2d, n: int, reinit: bool):
        super().__init__(n, reinit)
        self.members = nn.ModuleList([self._create_new_(module) for _ in range(n)])

        # copy the attributes of the bn layer
        self.num_features = module.num_features
        self.num_features = module.num_features
        self.eps = module.eps
        self.momentum = module.momentum
        self.affine = module.affine
        self.track_running_stats = module.track_running_stats

        factory_kwargs = {"device": None, "dtype": None}

        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(self.num_features, **factory_kwargs))
            self.register_buffer("running_var", torch.ones(self.num_features, **factory_kwargs))
            self.running_mean: Optional[Tensor]
            self.running_var: Optional[Tensor]
            self.register_buffer(
                "num_batches_tracked",
                torch.tensor(0, dtype=torch.long, **{k: v for k, v in factory_kwargs.items() if k != "dtype"}),
            )
            self.num_batches_tracked: Optional[Tensor]
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def get_weight(self):
        w = sum([self.members[i].weight * self.alpha[i] for i in range(self.n)])
        b = sum([self.members[i].bias * self.alpha[i] for i in range(self.n)])

        return w, b

    def __repr__(self):
        return "SubspaceBatchNorm2d(n={}, {num_features}, eps={eps}, momentum={momentum}, affine={affine}, track_running_stats={track_running_stats}".format(
            self.n, **self.__dict__
        )

    def retrieve_member(self, index):
        w = self.members[index].weight
        b = self.members[index].bias

        return w, b

    def retrieve_member_weight(self, index):
        return self.members[index].weight

    def forward(self, input):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()

        # The rest is code in the PyTorch source forward pass for batchnorm.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
