"""Tests for the dual-channel model registry integration."""

import torch


def test_dual_channel_model_registered_and_forward_shape() -> None:
    from sam_ml.modeling.models import get_model, list_models

    assert "dual_channel" in list_models()

    # Avoid downloading pretrained weights during tests.
    model = get_model("dual_channel", num_classes=5, use_pretrained=False)

    batch_size = 2
    x_clahe = torch.rand(batch_size, 3, 299, 299)
    x_ceced = torch.rand(batch_size, 3, 224, 224)

    y = model((x_clahe, x_ceced))
    assert y.shape == (batch_size, 5)

