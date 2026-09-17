import math
import torch
from torch.optim import Optimizer


class SGLD(Optimizer):
    r"""
    Stochastic Gradient Langevin Dynamics (SGLD) optimizer with temperature scaling
    and noise normalization based on dataset size.

    Updates parameters with Gaussian noise:
    θ ← θ - lr * grad + sqrt(2 * lr * temperature / num_samples) * N(0, 1)

    With momentum (gamma > 0) the SGHMC update (Wenzel et al. 2020 Alg. 1) is:
    p ← (1 - gamma) * p - lr * grad + sqrt(2 * gamma * temperature / num_samples) * N(0, I)
    θ ← θ + lr * p
    """

    def __init__(self, params, lr=0.001, weight_decay=0, temperature=1e-3, gamma=0.0):
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError(
                f"Invalid momentum / friction coefficient: {gamma}")

        defaults = dict(lr=lr, weight_decay=weight_decay,
                        temperature=temperature, gamma=gamma)
        super().__init__(params, defaults)

        # initialize momentum state
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['momentum'] = torch.zeros_like(p.data)

    @torch.no_grad()
    def step(self, num_samples):
        loss = None
        for group in self.param_groups:
            lr = group['lr']
            wd = group['weight_decay']
            temp = group['temperature']
            gamma = group['gamma']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)

                if gamma != 0:
                    state = self.state[p]
                    momentum = state['momentum']

                    # SGHMC (Chen et al. 2014, Wenzel et al. 2020 Alg. 1):
                    #   m ← (1 - γ) m - h ∇U + √(2γT) ξ,  θ ← θ + h m
                    # γ absorbs the continuous-time friction × step size, so
                    # the noise variance has no explicit lr factor.
                    momentum.mul_(1 - gamma).add_(grad, alpha=-lr)
                    noise_std = math.sqrt(2.0 * gamma * temp / num_samples)
                    momentum.add_(torch.randn_like(p.data) * noise_std)
                    p.data.add_(momentum, alpha=lr)
                else:
                    # gradient descent step
                    p.data.add_(grad, alpha=-lr)
                    # add Langevin noise
                    noise = torch.randn_like(p.data)
                    noise_std = math.sqrt(2.0 * lr * temp / num_samples)
                    p.data.add_(noise, alpha=noise_std)
        return loss


# Taken from: https://pysgmcmc.readthedocs.io/en/pytorch/_modules/pysgmcmc/optimizers/sgld.html#SGLD
# --------------------------------------------------------------------------------------------------
# Pytorch Port of a previous tensorflow implementation in `tensorflow_probability`:
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/g3doc/api_docs/python/tfp/optimizer/StochasticGradientLangevinDynamics.md


class SGLD2(Optimizer):
    """ Stochastic Gradient Langevin Dynamics Sampler with preconditioning.
        Optimization variable is viewed as a posterior sample under Stochastic
        Gradient Langevin Dynamics with noise rescaled in eaach dimension
        according to RMSProp.
    """

    def __init__(self,
                 params,
                 lr=1e-2,
                 precondition_decay_rate=0.95,
                 num_pseudo_batches=1,
                 num_burn_in_steps=3000,
                 diagonal_bias=1e-8) -> None:
        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        precondition_decay_rate : float, optional
            Exponential decay rate of the rescaling of the preconditioner (RMSprop).
            Should be smaller than but nearly `1` to approximate sampling from the posterior.
            Default: `0.95`
        num_pseudo_batches : int, optional
            Effective number of minibatches in the data set.
            Trades off noise and prior with the SGD likelihood term.
            Note: Assumes loss is taken as mean over a minibatch.
            Otherwise, if the sum was taken, divide this number by the batch size.
            Default: `1`.
        num_burn_in_steps : int, optional
            Number of iterations to collect gradient statistics to update the
            preconditioner before starting to draw noisy samples.
            Default: `3000`.
        diagonal_bias : float, optional
            Term added to the diagonal of the preconditioner to prevent it from
            degenerating.
            Default: `1e-8`.

        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if num_burn_in_steps < 0:
            raise ValueError(
                "Invalid num_burn_in_steps: {}".format(num_burn_in_steps))

        defaults = dict(
            lr=lr, precondition_decay_rate=precondition_decay_rate,
            num_pseudo_batches=num_pseudo_batches,
            num_burn_in_steps=num_burn_in_steps,
            diagonal_bias=1e-8,
        )
        super().__init__(params, defaults)

    def step(self, num_samples=None, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr = group["lr"]
                num_pseudo_batches = group["num_pseudo_batches"]
                precondition_decay_rate = group["precondition_decay_rate"]
                gradient = parameter.grad.data

                if len(state) == 0:
                    state["iteration"] = 0
                    state["momentum"] = torch.ones_like(parameter)

                state["iteration"] += 1

                momentum = state["momentum"]

                #  Momentum update {{{ #
                momentum.add_(
                    (1.0 - precondition_decay_rate) *
                    ((gradient ** 2) - momentum)
                )
                #  }}} Momentum update #

                if state["iteration"] > group["num_burn_in_steps"]:
                    sigma = 1. / torch.sqrt(torch.tensor(lr))
                else:
                    sigma = torch.zeros_like(parameter)

                preconditioner = (
                    1. / torch.sqrt(momentum + group["diagonal_bias"])
                )

                scaled_grad = (
                    0.5 * preconditioner * gradient * num_pseudo_batches +
                    torch.normal(
                        mean=torch.zeros_like(gradient),
                        std=torch.ones_like(gradient)
                    ) * sigma * torch.sqrt(preconditioner)
                )

                parameter.data.add_(-lr * scaled_grad)

        return loss
