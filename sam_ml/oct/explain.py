"""Occlusion sensitivity (paper method) and complementary Grad-CAM."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@torch.inference_mode()
def occlusion_sensitivity(
    model, image: torch.Tensor, target_class: int | None = None,
    window: int = 28, stride: int = 7, value: float = 0.0,
) -> torch.Tensor:
    """Measure probability decrease after sliding square occlusion over a BCHW image."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Expected a single BCHW image")
    model.eval()
    baseline = torch.softmax(model(image), dim=1)[0]
    target = int(baseline.argmax()) if target_class is None else target_class
    _, _, height, width = image.shape
    heat = torch.zeros((height, width), device=image.device)
    counts = torch.zeros_like(heat)
    for y in range(0, max(1, height - window + 1), stride):
        for x in range(0, max(1, width - window + 1), stride):
            altered = image.clone()
            altered[:, :, y:y + window, x:x + window] = value
            probability = torch.softmax(model(altered), dim=1)[0, target]
            heat[y:y + window, x:x + window] += baseline[target] - probability
            counts[y:y + window, x:x + window] += 1
    return heat / counts.clamp_min(1)


def gradcam(model, image: torch.Tensor, target_layer, target_class: int | None = None) -> torch.Tensor:
    """Compute Grad-CAM separately from the paper-reproduction occlusion method."""
    activations, gradients = [], []
    forward_hook = target_layer.register_forward_hook(lambda _m, _i, output: activations.append(output))
    backward_hook = target_layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
    try:
        logits = model(image)
        target = int(logits.argmax(1)) if target_class is None else target_class
        model.zero_grad(set_to_none=True)
        logits[0, target].backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return cam / cam.max().clamp_min(1e-12)
    finally:
        forward_hook.remove()
        backward_hook.remove()


def save_heatmap_triplet(original: torch.Tensor, heatmap: torch.Tensor, prefix: str) -> None:
    """Save original, normalized map, and red overlay without matplotlib."""
    base = original.detach().cpu()
    if base.ndim == 4:
        base = base[0]
    base = base.mean(0).numpy()
    base = (base - base.min()) / max(float(base.max() - base.min()), 1e-12)
    heat = heatmap.detach().cpu().numpy()
    heat = (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-12)
    gray = (base * 255).astype(np.uint8)
    heat_u8 = (heat * 255).astype(np.uint8)
    overlay = np.stack([gray, gray, gray], axis=-1).astype(float)
    overlay[..., 0] = 0.6 * overlay[..., 0] + 0.4 * heat_u8
    Image.fromarray(gray).save(f"{prefix}_original.png")
    Image.fromarray(heat_u8).save(f"{prefix}_heatmap.png")
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(f"{prefix}_overlay.png")
