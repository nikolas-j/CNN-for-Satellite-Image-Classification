# Logic for data validation and prediction using the ML model

import torch
from app.ml.models.cnn_ship_classifier_model import CNNShipClassifier
import torch.nn.functional as F
import io
import torchvision.transforms as transforms
from PIL import Image

from app.ml.utils import make_predictions


class ShipClassifier:
    def __init__(self, model_path: str, class_names: list):
        """
        Initializes the ShipClassifier with the given model path and class names.
        Loads the model weights from the specified path.
        """

        self.class_names = class_names
        self.model = CNNShipClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((80, 80)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Called from endpoints.py
    # Takes in image bytes from uploaded file
    # Returns predicted label and confidence score
    def predict(self, image_bytes: bytes):

        # Preprocess the png image into [3x80x80] tensor for the model
        image = Image.open(io.BytesIO(image_bytes))
        image = self.transform(image)

        images = [image.squeeze()] # Placeholder for single image list

        # Make prediction using utility function which calls the model
        # Can be extended to batch predictions in the future
        # Currently returns single prediction label and probability in a list
        pred_label, pred_prob = make_predictions(self.model, images)

        # Modify later
        return pred_label[0], pred_prob[0]