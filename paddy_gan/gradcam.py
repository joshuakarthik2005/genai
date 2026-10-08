"""Grad-CAM heatmap visualization for the ResNet-18 disease classifier.

Shows which regions of a leaf image the model uses to make its prediction.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .classifier import eval_tf


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ResNet-18."""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self._features = None
        self._grads = None
        # Hook onto the last conv layer of ResNet-18 (layer4)
        self._fwd_hook = model.layer4.register_forward_hook(self._save_features)
        self._bwd_hook = model.layer4.register_full_backward_hook(self._save_grads)

    def _save_features(self, module, input, output):
        self._features = output.detach()

    def _save_grads(self, module, grad_input, grad_output):
        self._grads = grad_output[0].detach()

    def __call__(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            img_tensor: [1, 3, H, W] normalized tensor (from eval_tf)
            class_idx:  class to explain. None = argmax (predicted class).

        Returns:
            cam (np.ndarray [H, W] in [0,1]), predicted_class_idx (int), logits
        """
        device = next(self.model.parameters()).device
        x = img_tensor.to(device)
        x.requires_grad_(True)

        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool the gradients over spatial dims
        weights = self._grads.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self._features).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)
        cam = cam[0, 0].cpu().numpy()

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        # Upsample to input size
        h, w = img_tensor.shape[2], img_tensor.shape[3]
        cam_tensor = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(cam_tensor, size=(h, w), mode="bilinear", align_corners=False)
        cam_up = cam_up[0, 0].numpy()

        return cam_up, class_idx, logits[0].detach().cpu()

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a CAM heatmap (jet colormap) on the original image."""
    import matplotlib.cm as cm

    colormap = cm.get_cmap("jet")
    heatmap = colormap(cam)[:, :, :3]
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_uint8).resize(image.size, Image.BILINEAR)

    img_arr = np.array(image.convert("RGB")).astype(float)
    heat_arr = np.array(heatmap_pil).astype(float)
    blended = (img_arr * (1 - alpha) + heat_arr * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def explain_prediction(net, classes: list, img_size: int, image: Image.Image,
                        class_idx: int = None):
    """Run Grad-CAM on a single image and return overlay + metadata.

    Returns:
        dict with keys: overlay (PIL), cam (np.ndarray), pred_class, pred_prob, all_probs
    """
    tf = eval_tf(img_size)
    x = tf(image.convert("RGB")).unsqueeze(0)

    gcam = GradCAM(net)
    cam, pred_idx, logits = gcam(x, class_idx)
    gcam.remove_hooks()

    probs = torch.softmax(logits, dim=0).tolist()
    overlay = overlay_heatmap(image.resize((img_size, img_size)), cam)

    return {
        "overlay": overlay,
        "cam": cam,
        "pred_class": classes[pred_idx],
        "pred_prob": probs[pred_idx],
        "all_probs": list(zip(classes, probs)),
    }
