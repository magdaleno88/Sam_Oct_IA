"""Memory-safe probability ensemble utilities."""

from collections.abc import Iterable

import torch


@torch.inference_mode()
def average_probabilities(logits: Iterable[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return arithmetic mean and per-class population std of model probabilities."""
    probabilities = [torch.softmax(item.detach().cpu(), dim=-1) for item in logits]
    if not probabilities:
        raise ValueError("At least one model output is required")
    stacked = torch.stack(probabilities)
    return stacked.mean(dim=0), stacked.std(dim=0, unbiased=False)


@torch.inference_mode()
def sequential_ensemble_predict(models, inputs: torch.Tensor, device: torch.device | str = "cpu"):
    """Run one model at a time and accumulate probabilities on CPU."""
    outputs = []
    for model in models:
        model = model.to(device).eval()
        outputs.append(model(inputs.to(device)).detach().cpu())
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    mean, std = average_probabilities(outputs)
    entropy = -(mean.clamp_min(1e-12) * mean.clamp_min(1e-12).log()).sum(dim=-1)
    return {"probabilities": mean, "std": std, "entropy": entropy, "logits": outputs}
