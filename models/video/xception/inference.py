import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm

# =========================
# MODEL DEFINITION
# =========================
def build_xceptionnet(num_classes=2, dropout=0.5):
    model = timm.create_model('xception', pretrained=False, num_classes=num_classes)

    feature_dim = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(feature_dim, num_classes)
    )
    return model


# =========================
# DETECTOR CLASS
# =========================
class XceptionDetector:
    def __init__(self, weight_path, device="cuda"):
        self.device = device

        self.model = build_xceptionnet().to(device)
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.eval()

        # 🔥 EXACT SAME AS TRAINING (val_transform)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
        ])

    def predict_frame(self, frame):
        """
        frame: numpy array (H, W, C)
        """
        img = self.transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img)
            prob = torch.softmax(output, dim=1)

        return prob.cpu().numpy()[0]  # [real_prob, fake_prob]