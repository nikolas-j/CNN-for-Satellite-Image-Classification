# Helper functions for ML tasks

import torch

# Predictions function for a list of images in tensor form [3, 80, 80] normalized to 0-1 pixel values
# Returns list of prediction labels and list of prediction probabilities
def make_predictions(model: torch.nn.Module, data: list):
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