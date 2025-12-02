import torch.nn as nn
import torch


class TLU(nn.Module):
    def __init__(self, num_features):
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_eps_learnable=False):
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_learnable = is_eps_learnable

        self.weight = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_learnable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_learnable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        x = x * torch.rsqrt(nu2 + self.eps.abs())

        x = self.weight * x + self.bias
        return x


class PackedTLU(nn.Module):
    def __init__(self, num_features, num_estimators=4, alpha=2, gamma=1):
        super(PackedTLU, self).__init__()
        self.num_estimators = num_estimators
        self.alpha = alpha
        self.gamma = gamma

        assert num_features % num_estimators == 0
        self.num_features = int(num_features / num_estimators) * alpha

        self.tlu_list = nn.ModuleList([
            TLU(self.num_features) for _ in range(num_estimators)
        ])

    def forward(self, x):
        x = torch.chunk(x, self.num_estimators, dim=1)

        out = []
        for i, tlu in enumerate(self.tlu_list):
            out.append(tlu(x[i]))

        return torch.cat(out, dim=1)


class PackedFRN(nn.Module):
    def __init__(self, num_features, num_estimators=4, alpha=2, gamma=1, eps=1e-6, is_eps_learnable=False):
        super(PackedFRN, self).__init__()
        self.num_estimators = num_estimators
        self.alpha = alpha
        self.gamma = gamma

        assert num_features % num_estimators == 0
        self.num_features = int(num_features / num_estimators) * alpha

        self.frn_list = nn.ModuleList([
            FRN(self.num_features, eps=eps, is_eps_learnable=is_eps_learnable)
            for _ in range(num_estimators)
        ])

    def forward(self, x):
        x = torch.chunk(x, self.num_estimators, dim=1)

        out = []
        for i, frn in enumerate(self.frn_list):
            out.append(frn(x[i]))

        return torch.cat(out, dim=1)
