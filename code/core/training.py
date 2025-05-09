import numpy as np
import pandas as pd
import torch

from sklearn.metrics import classification_report


def compute_smoothed_weights(class_counts, alpha=0.5):
    """
    Computes smoothed class weights for imbalanced datasets.

    Parameters:
        class_counts (list or numpy.ndarray): Array of class counts.
        alpha (float, optional): Smoothing factor. Higher values give more weight to smaller classes.
                                 Defaults to 0.5.

    Returns:
        numpy.ndarray: Normalized class weights.
    """
    class_counts = np.array(class_counts, dtype=np.float32)
    weights = (1.0 / class_counts) ** alpha
    return weights / weights.sum()  

def pred_val_data(model, data_loader):
    """
    Generates predictions and ground truth masks for a validation dataset.

    Parameters:
        model (torch.nn.Module): The trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Predicted class indices for the validation dataset.
            - numpy.ndarray: Ground truth masks for the validation dataset.
    """
    model.eval()

    pred_list = []
    mask_list = []
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(model.device)
            outputs = model(images)
            output_classes = torch.argmax(outputs, dim=1)
            pred_list.append(output_classes.cpu().numpy())
            mask_list.append(masks)

    # Concatenate the predictions and labels along the batch dimension
    preds = np.concatenate(pred_list, axis=0)
    masks = np.concatenate(mask_list, axis=0)

    return preds, masks

def calc_class_report(y_pred, y_true, ignore_index, class_mapping):
    """
    Calculates a classification report for the predicted and true labels.

    Parameters:
        y_pred (numpy.ndarray): Predicted class indices.
        y_true (numpy.ndarray): Ground truth class indices.
        ignore_index (int): Index to ignore in the evaluation (e.g., background or no-legend class).
        class_mapping (dict): Mapping of class indices to class names.

    Returns:
        pandas.DataFrame: A DataFrame containing the classification report, including precision, recall,
                          F1-score, and support for each class.
    """
    idx_no_legend = y_true.flatten() != ignore_index
    class_names = [val for key, val in sorted(class_mapping.items()) if key != ignore_index]
    class_report = classification_report(
        y_true.flatten()[idx_no_legend],
        y_pred.flatten()[idx_no_legend],
        target_names=class_names,
        output_dict=True)
    
    class_report_df = pd.DataFrame(class_report).transpose()

    return class_report_df