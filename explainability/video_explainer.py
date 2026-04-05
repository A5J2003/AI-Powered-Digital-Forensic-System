import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ==========================================
# DEVICE
# ==========================================
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# MODELS (MATCH TRAINING EXACTLY)
# ==========================================
def load_xception(weights_path, device):
    model = timm.create_model('xception', pretrained=False, num_classes=2)

    feature_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(feature_dim, 2)
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


class VideoSwinDeepfake(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ MUST MATCH TRAINING (global_pool='avg')
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )

        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, 2)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features = self.backbone(x)
        features = features.reshape(B, T, -1).mean(dim=1)
        return self.head(features)


def load_swin(weights_path, device):
    model = VideoSwinDeepfake()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    return model


# ==========================================
# TRANSFORMS
# ==========================================
def get_xception_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])


def get_swin_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])


# ==========================================
# 🔥 ATTENTION ROLLOUT (REAL ATTENTION)
# ==========================================
class AttentionRollout:
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for layer in self.model.backbone.layers:
            for block in layer.blocks:
                hook = block.attn.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)

    def _hook_fn(self, module, input, output):
        with torch.no_grad():
            B_, N, C = input[0].shape
            qkv = module.qkv(input[0])
            qkv = qkv.reshape(B_, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv.unbind(0)
            scale = (C // module.num_heads) ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            self.attention_maps.append(attn.detach().cpu())

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __call__(self, input_tensor):
        self.attention_maps = []
        with torch.no_grad():
            self.model(input_tensor)

        if len(self.attention_maps) == 0:
            return np.ones((7, 7), dtype=np.float32)

        processed = []
        for attn in self.attention_maps:
            attn_avg = attn.mean(dim=1)
            eye = torch.eye(attn_avg.size(-1)).unsqueeze(0)
            attn_avg = attn_avg + eye
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            processed.append(attn_avg[0])

        rollout = processed[0]
        for attn in processed[1:]:
            n = min(attn.shape[0], rollout.shape[0])
            rollout = torch.matmul(attn[:n, :n], rollout[:n, :n])

        attn_map = rollout.mean(dim=0).numpy()
        grid_size = int(np.sqrt(attn_map.shape[0]))
        attn_map = attn_map[:grid_size * grid_size].reshape(grid_size, grid_size)
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map


# ==========================================
# SAVE GRID
# ==========================================
def save_gradcam_grid(images, save_path):
    cols = 4
    rows = (len(images) + cols - 1) // cols

    plt.figure(figsize=(12, 3 * rows))

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# ==========================================
# 🔥 MAIN (FULLY FIXED)
# ==========================================
def run_explainability(frames_folder, case, device=None):

    print("\n🔥 ENTERED run_explainability()")

    device = get_device() if device is None else device

    frame_files = sorted([
        os.path.join(frames_folder, f)
        for f in os.listdir(frames_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if len(frame_files) == 0:
        return {}

    indices = np.linspace(0, len(frame_files)-1, 16, dtype=int)

    frames = []
    for idx in indices:
        img = cv2.imread(frame_files[idx])
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    explain_dir = case.get_path("explain")
    os.makedirs(explain_dir, exist_ok=True)

    gradcam_path = os.path.join(explain_dir, "video_gradcam_grid.png")
    attention_path = os.path.join(explain_dir, "video_swin_attention.png")

    model_x = load_xception("models/video/xception/xceptionnet_best.pth", device)
    model_s = load_swin("models/video/swin/swin_best.pth", device)

    tx = get_xception_transform()
    ts = get_swin_transform()

    print("🔥 Running GradCAM...")

    visualizations = []
    frame_scores = []   # ✅ NEW

    for frame in frames:
        resized = cv2.resize(frame, (299, 299))
        img_float = resized.astype(np.float32) / 255.0

        input_tensor = tx(Image.fromarray(resized)).unsqueeze(0).to(device)

        # ✅ NEW: frame score
        with torch.no_grad():
            output = model_x(input_tensor)
            probs = torch.softmax(output, dim=1)
            fake_prob = probs[0, 1].item()
            frame_scores.append(fake_prob)

        with GradCAM(model=model_x, target_layers=[model_x.conv4]) as cam:
            cam_map = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(1)])[0]

        viz = show_cam_on_image(img_float, cam_map, use_rgb=True)
        visualizations.append(viz)

    save_gradcam_grid(visualizations, gradcam_path)
    print("✅ GradCAM saved")

    # =========================
    # SWIN ATTENTION (UNCHANGED)
    # =========================
    print("🔥 Running Swin Attention...")

    try:
        class SingleFrameModel(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.backbone = m.backbone

            def forward(self, x):
                return self.backbone(x)

        rollout = AttentionRollout(SingleFrameModel(model_s).to(device))

        attn_maps = []

        for frame in frames:
            resized = cv2.resize(frame, (224, 224))
            img_float = resized.astype(np.float32) / 255.0

            input_tensor = ts(Image.fromarray(resized)).unsqueeze(0).to(device)

            attn_map = rollout(input_tensor)
            attn_resized = cv2.resize(attn_map, (224, 224))

            viz = show_cam_on_image(img_float, attn_resized, use_rgb=True)
            attn_maps.append(viz)

        rollout.remove_hooks()

        save_gradcam_grid(attn_maps, attention_path)
        print("✅ Attention saved")

    except Exception as e:
        print("⚠️ Attention failed:", e)

    return {
        "gradcam_path": gradcam_path,
        "attention_path": attention_path,

        # ✅ NEW (for timestamps)
        "details": {
            "frame_scores": frame_scores
        }
    }