import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import timm

# =========================
# MODEL
# =========================
class VideoSwinDeepfake(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()

        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )

        feature_dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        features = self.backbone(x)
        features = features.reshape(B, T, -1).mean(dim=1)

        return self.head(features)


# =========================
# DETECTOR
# =========================
class SwinDetector:
    def __init__(self, weight_path, device="cuda", num_frames=16):
        self.device = device
        self.num_frames = num_frames

        self.model = VideoSwinDeepfake().to(device)
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    # =========================
    # FRAME SAMPLING
    # =========================
    def sample_frames(self, frames):
        n = len(frames)

        if n >= self.num_frames:
            indices = np.linspace(0, n - 1, self.num_frames, dtype=int)
            return [frames[i] for i in indices]
        else:
            sampled = frames.copy()
            while len(sampled) < self.num_frames:
                sampled.extend(frames[:self.num_frames - len(sampled)])
            return sampled

    # =========================
    # PREDICT VIDEO
    # =========================
    def predict_video(self, frames):
        """
        frames: list of numpy arrays
        """

        frames = self.sample_frames(frames)

        processed = []
        for frame in frames:
            img = self.transform(frame)
            processed.append(img)

        # [T, C, H, W] → [C, T, H, W]
        video_tensor = torch.stack(processed).permute(1, 0, 2, 3)
        video_tensor = video_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(video_tensor)
            prob = torch.softmax(output, dim=1)

        return prob.cpu().numpy()[0]  # [real, fake]