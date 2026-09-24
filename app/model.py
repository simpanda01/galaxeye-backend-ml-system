import torch
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = "satellite_model.pth"


class SatelliteClassifier:

    def __init__(self):
        self.device = torch.device("cpu")

        # Load the saved model checkpoint
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device
        )

        self.classes = checkpoint["classes"]

        # Create the same ResNet18 architecture used during training
        self.model = models.resnet18(weights=None)

        self.model.fc = torch.nn.Linear(
            self.model.fc.in_features,
            len(self.classes)
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        # Same preprocessing used during training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image: Image.Image):

        image = image.convert("RGB")

        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)

        confidence, class_index = torch.max(
            probabilities,
            dim=1
        )

        prediction = self.classes[class_index.item()]

        return {
            "prediction": prediction,
            "confidence": round(confidence.item(), 4)
        }