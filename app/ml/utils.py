# Helper functions for ML tasks

import torch
from PIL import Image
import torchvision.transforms as transforms
import io
from sklearn.cluster import DBSCAN
import numpy as np

def preprocess_image(image, resolution_m_per_pixel: float):
    '''
    Preprocess the input image for the model.
    Converts bytes to a PIL Image
    Resizes to match training resolution (3 m / pixel)
    Normalizes pixel values [0,1]
    '''

    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image)).convert("RGB")

    # Resize image to 3 m / pixel resolution used by model
    target_resolution = 3.0  # meters per pixel
    scale_factor = resolution_m_per_pixel / target_resolution
    new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
    image = image.resize(new_size, Image.BILINEAR)

    # Normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image)

    return image


def tile_image(image_tensor: torch.Tensor, tile_size, stride):
    '''
    Tiles the landscape image into overlapping [80x80] patches for prediction.
    Returns a list of image tensors and a list of tuples as positions in the tile grid.
    '''
    images = []
    positions = []
    _, height, width = image_tensor.shape

    if height < tile_size or width < tile_size:
        raise ValueError("Image is smaller than the tile size.")

    # Calculate number of tiles in x and y directions
    max_tile_index_x = (width-tile_size) // stride
    max_tile_index_y = (height-tile_size) // stride

    # Loop over the image to create tiles
    # Note: We discard tiles that would go beyond image boundaries
    # resulting in a max loss of stride-1 pixels on rigth and bottom edge

    for y in range(0, max_tile_index_y + 1):
        for x in range(0, max_tile_index_x + 1):
            x_index = x*stride
            y_index = y*stride
            image = image_tensor[:, y_index:(y_index + tile_size), x_index:(x_index + tile_size)]
            images.append(image)
            positions.append((x,y))

    return images, positions


def cluster_positions(pred_labels: list, image_positions: list, distance_threshold: float):
    '''
    Clusters predicted ship positions to avoid multiple detections of the same ship using DBSCAN.
    '''
    clustered_positions = []
    clustered_count = 0

    # Extract positions where ships were detected
    ship_positions = [pos for label, pos in zip(pred_labels, image_positions) if label == 1]

    # Skip clustering if a single or no ship detected
    if len(ship_positions) == 0:
        return 0, [(0.0, 0.0)]
    if len(ship_positions) == 1:
        offset = 0.5  # Center of the tile
        avg_x = (ship_positions[0][0] + offset) / (image_positions[-1][0] + 1)
        avg_y = (ship_positions[0][1] + offset) / (image_positions[-1][1] + 1)
        return 1, [(avg_x, avg_y)]

    # Find clusters
    # Convert to numpy array for DBSCAN
    ship_positions = np.array(ship_positions)
    dbscan = DBSCAN(eps=distance_threshold, min_samples=1)
    clusters = dbscan.fit(ship_positions)
    labels = clusters.labels_
    unique_labels = set(labels)
    clustered_count = int(len(unique_labels)) # Ensure python int type for JSON serialization

    # Calculate average position for each cluster, relative to image size [0,1]
    for label in unique_labels:
        cluster_points = [ship_positions[i] for i in range(len(ship_positions)) if labels[i] == label]
        offset = 0.5  # Center of the tile
        avg_x = (sum([point[0] for point in cluster_points]) / len(cluster_points) + offset) / (image_positions[-1][0] + 1)
        avg_y = (sum([point[1] for point in cluster_points]) / len(cluster_points) + offset) / (image_positions[-1][1] + 1)

        clustered_positions.append((avg_x.item(), avg_y.item())) # Ensure python float type for JSON serialization

    return clustered_count, clustered_positions


def make_predictions(model: torch.nn.Module, data: list):
    '''
    Predictions function for a list of images in tensor form [3, 80, 80] normalized to 0-1 pixel values
    Returns list of prediction labels and list of prediction probabilities
    '''
    pred_labels = []
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0) # Add an extra dimension for batch size of 1
            # Forward pass (model outputs raw logit for each class)
            pred_logit = model(sample)
            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            # note: perform softmax on the "logits" dimension, not "batch" dimension
            # (in this case we have a batch size of 1, so can perform on dim=0)
            pred_probs.append(pred_prob.max(dim=0).values)  # Store max probability for the predicted class
            pred_labels.append(pred_prob.argmax(dim=0).item())  # Store predicted class label

    return pred_labels, pred_probs