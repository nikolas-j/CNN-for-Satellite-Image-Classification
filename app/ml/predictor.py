# Logic for data validation and prediction using the ML model

import torch
from app.ml.models.cnn_ship_classifier_model import CNNShipClassifier
from app.ml.utils import make_predictions, tile_image, preprocess_image, cluster_positions

class ShipClassifier:
    def __init__(self, model_path: str = "app/ml/models/model_weights_v1.pth"):
        """
        Initializes the ShipClassifier with the given model path.
        Loads the model weights from the specified path.
        """
        
        self.model = CNNShipClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.TILE_SIZE = 80  # Depends on model input size
        self.STRIDE = 15     # Stride for tiling -> Trade off between speed and accuracy
        self.DISTANCE_THRESHOLD = 150  # meters for clustering positions

    # Called from endpoints.py
    # Takes in image bytes from uploaded file
    # Returns predicted label and confidence score
    def predict(self, image_bytes: bytes, resolution_m_per_pixel: float):

        # Preprocess the png image into [3x80x80] tensor for the model
        image_scaled_normalized = preprocess_image(image_bytes, resolution_m_per_pixel)

        # Tile the image into overlapping tiles
        images, image_positions = tile_image(image_scaled_normalized, tile_size=self.TILE_SIZE, stride=self.STRIDE) 

        # Make prediction
        # pred_prob can be used for confidence scores if needed
        pred_label, pred_prob = make_predictions(self.model, images)

        # Cluster positions to avoid multiple detections of the same ship
        ship_count, positions = cluster_positions(pred_label, 
                                                  image_positions, 
                                                  self.DISTANCE_THRESHOLD / (resolution_m_per_pixel * self.STRIDE))

        # return ship count and positions
        return ship_count, positions