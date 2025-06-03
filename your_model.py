import logging
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
from io import BytesIO
import torch.nn.functional as F

class MetaMLP(nn.Module):
    def __init__(self, input_size=7):  # 3 CLIP + 2 Swin + 2 ResNet
        super(MetaMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 класса

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ModelInf:
    def __init__(self, use_cuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Используется устройство: {self.device}")
        resnet_path = "models/resnet50_model_checkpoint.pth"
        self.resnet = torch.load(resnet_path, weights_only=False, map_location=self.device)
        self.resnet = self.resnet.to(self.device)
        self.resnet.eval()
        for param in self.resnet.parameters():
            param.requires_grad = False

        # === Swin Transformer ===
        swin_path = "models/swin-t_model_checkpoint.pth"
        self.swin = torch.load(swin_path, weights_only=False, map_location=self.device)
        self.swin = self.swin.to(self.device)
        self.swin.eval()
        for param in self.swin.parameters():
            param.requires_grad = False

        # === CLIP ===
        self.clip_model = CLIPModel.from_pretrained("models/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("models/clip-vit-base-patch32")
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_prompts = ["AI generated image", "Human-made photo", "Human-made computer art"]

        self.meta_model = MetaMLP(input_size=7)
        self.meta_model.load_state_dict(
            torch.load("models/meta_mlp_weights.pth", weights_only=False, map_location="cpu"))
        self.meta_model.to(self.device)
        self.meta_model.eval()

    def predict_image_with_ensemble(self, image_bytes):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        # === 1. Предобработка
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(self.device)  # [1, 3, 256, 256]

        # === 2. ResNet вывод
        with torch.no_grad():
            resnet_probs = F.softmax(self.resnet(image_tensor), dim=1).squeeze()  # [2]
            resnet_feat = resnet_probs  # только два класса

        # === 3. Swin вывод
        with torch.no_grad():
            swin_probs = F.softmax(self.swin(image_tensor), dim=1).squeeze()  # [2]
            swin_feat = swin_probs[:2]

        # === 4. CLIP
        with torch.no_grad():
            inputs = self.clip_processor(text=self.clip_prompts, images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.clip_model(**inputs)
            clip_probs = outputs.logits_per_image.softmax(dim=1).squeeze()  # [3]
            clip_feat = clip_probs  # [3]

        # === 5. Объединение фичей
        features = torch.cat([resnet_feat, swin_feat, clip_feat], dim=0).unsqueeze(0).to(self.device)  # [1, 7]

        # === 6. Meta-MLP
        with torch.no_grad():
            pred = self.meta_model(features)
            pred_class = torch.argmax(pred, dim=1).item()
            pred_probs = F.softmax(pred, dim=1).squeeze().cpu().numpy()

        return {
            "class_id": pred_class,
            "class_name": ["AI-generated", "Not AI"][pred_class],
            "meta_probs": pred_probs,
            "clip_probs": clip_probs.cpu().numpy(),
            "resnet_probs": resnet_probs.cpu().numpy(),
            "swin_probs": swin_probs.cpu().numpy()
        }
        """
        return f"Class: {["AI-generated", "Not AI"][pred_class]}" """
    '''
    def predict(self, image_bytes):
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.test_transforms(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_batch)
        prob, predicted_class = torch.max(torch.softmax(output, dim=1), 1)

        if predicted_class.item() == 1:
            label = "Not AI"
        else:
            label = "AI"

        confidence = f"{prob.item() * 100:.2f}%"
        return f"{label} ({confidence})"
    '''
    # Legacy, may be used for improved performance on deployment
    # However cannot be used on rasberry pi 2b (could not install onnx runtime)
    '''def to_onnx(self, filename:str="model"):
        dummy_input = torch.randn(1, 3, 256, 256)
        torch.onnx.export(
            self.model,
            (dummy_input,),
            filename + ".onnx",
            input_names=["input"],
            output_names=["output"],
            opset_version=13
        )'''


# Инициализация модели один раз при импорте
_model_inf = ModelInf()
# тут надо редачить и создавать отдельный файл с подгрузкой onnx
def predict_is_ai(image_bytes):
    outp = _model_inf.predict_image_with_ensemble(image_bytes)
    swin_pred = ["AI-generated", "Not AI"][np.argmax(outp["swin_probs"])]
    resnet_pred = ["AI-generated", "Not AI"][np.argmax(outp["resnet_probs"])]
    clip_pred = ["AI generated image", "Human-made photo", "Human-made computer art"][
        np.argmax(outp["resnet_probs"])]
    clip_class_probs = np.array([outp["clip_probs"][0], outp["clip_probs"][1] + outp["clip_probs"][2]])
    sum_decision = ["AI-generated", "Not AI"][np.argmax(outp["swin_probs"] + outp["resnet_probs"] + clip_class_probs)]
    return f"\nSwin predict: {swin_pred}\nResnet predict: {resnet_pred}\nClip predict: {clip_pred}\nSum predict: {sum_decision}\nMLP predict: {outp["class_name"]}"