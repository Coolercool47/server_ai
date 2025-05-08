import logging
from PIL import Image
import torch
#import torch.onnx
from torchvision import transforms
from io import BytesIO

class ModelInf:
    def __init__(self, use_cuda=True):
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            device_name = "GPU"
        else:
            self.device = torch.device("cpu")
            device_name = "CPU"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.info(f"Используется устройство: {device_name}")
        if device_name == "CPU":
            self.model = torch.load("model_checkpoint.pth", weights_only=False, map_location=torch.device('cpu'))
        else:
            self.model = torch.load("model_checkpoint.pth", weights_only=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.test_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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
    return _model_inf.predict(image_bytes)