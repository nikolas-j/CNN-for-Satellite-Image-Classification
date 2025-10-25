# Unit tests for ml predictor and utils

import pytest
import torch
from app.main import app
from app.ml.predictor import ShipClassifier
from app.ml.utils import tile_image, cluster_positions

def test_tile_image():
    # Create a dummy image tensor of size [3x200x200]
    image_tensor = torch.randn(3, 200, 200)
    tile_size = 80
    stride = 20

    images, positions = tile_image(image_tensor, tile_size, stride)

    # Calculate expected number of tiles
    expected_num_tiles_x = (200 - tile_size) // stride + 1
    expected_num_tiles_y = (200 - tile_size) // stride + 1
    expected_num_tiles = expected_num_tiles_x * expected_num_tiles_y

    assert len(images) == expected_num_tiles
    assert len(positions) == expected_num_tiles
    for img in images:
        assert img.shape == (3, tile_size, tile_size)

def test_cluster_positions():
    # Sample predicted labels and positions
    pred_labels = [1, 1, 0, 1, 1, 0]
    image_positions = [(10, 10), (12, 11), (50, 50), (11, 13), (13, 12), (80, 80)]
    distance_threshold = 5.0  # Small threshold to cluster close points

    ship_count, ship_positions = cluster_positions(pred_labels, image_positions, distance_threshold)

    # We expect the first four positions to cluster into one ship
    assert ship_count == 1
    assert len(ship_positions) == 1
    avg_x = (10 + 12 + 11 + 13) / 4 + 0.5
    avg_y = (10 + 11 + 13 + 12) / 4 + 0.5
    assert ship_positions[0] == (avg_x / (image_positions[-1][0] + 1), avg_y / (image_positions[-1][1] + 1))


def test_ship_classifier_predict():

    classifier = ShipClassifier("tests/model_test_weights_v1.pth")

    # Create a dummy image bytes (a simple black square PNG)
    from PIL import Image
    import io

    img = Image.new('RGB', (200, 200), color = 'black')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    image_bytes = img_byte_arr.getvalue()

    resolution_m_per_pixel = 3.0

    ship_count, positions = classifier.predict(image_bytes, resolution_m_per_pixel)

    # Since the image is blank, we expect zero ships detected
    assert ship_count == 0
    assert positions == [(0.0, 0.0)]

