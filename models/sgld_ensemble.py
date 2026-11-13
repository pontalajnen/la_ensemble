import torch
import torch.nn as nn
from copy import deepcopy


class SGLDEnsemble(nn.Module):
    """Wraps a collection of SGLD posterior samples for Bayesian model averaging.

    At inference, each sample model produces a softmax distribution; the
    ensemble returns their mean — an unweighted Monte Carlo estimate of the
    posterior predictive.
    """

    def __init__(self, models: list):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return log of the averaged posterior-predictive probabilities.
        # Downstream eval code applies torch.softmax(output, dim=-1); feeding
        # it log-probs recovers probs exactly (softmax is invariant to an
        # additive shift and the probs already sum to 1 per row).
        preds = torch.stack(
            [torch.softmax(m(x), dim=-1) for m in self.models], dim=0
        )
        mean_probs = preds.mean(dim=0).clamp(min=1e-12)
        return mean_probs.log()

    @property
    def num_samples(self) -> int:
        return len(self.models)

    @classmethod
    def from_checkpoints(cls, checkpoint_paths: list, model_factory, device, **load_kwargs):
        """Load one checkpoint per path using *model_factory()* and return an SGLDEnsemble.

        Args:
            checkpoint_paths: ordered list of .pth file paths.
            model_factory: zero-arg callable that returns a fresh model instance.
            device: torch device to move each model to.
            **load_kwargs: passed to torch.load (e.g. weights_only=True).
        """
        models = []
        for path in checkpoint_paths:
            m = model_factory()
            state = torch.load(path, map_location=device, **load_kwargs)
            m.load_state_dict(state)
            m.to(device)
            m.eval()
            models.append(m)
        return cls(models)
