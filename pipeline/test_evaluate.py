# evaluate.py
# Evaluate a trained SlideClassifier model on test or subset data.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.create_datasets import SlideDataset
from wsi_models.wsi_models import SlideClassifier
import yaml
import numpy as np


# helper function to computed all required metrices
# accuracy, precision, recall and F1
# to avoid sklearn metrics, because was not installed into the docker

def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    acc = (y_true == y_pred).mean()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    prec = tp / (tp + fp + 1e-8) # 1e-8 to prevent crashes
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)

    return acc, prec, rec, f1


def run_evaluation(config, model_path="outputs/slide_classifier.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------
    # Dataset (test or subset)
    # -------------------------------
    if config["datasets"]["use_subset"]:
        print("[INFO] Evaluating SUBSET dataset (entire subset, no fraction)")
        test_set = SlideDataset(
            metadata_csv=config["data"]["metadata_csv"],
            organ=config["data"]["organ"],
            subset_csv=config["datasets"]["subset"],
            features_dir=config["features"]["dir"],
        )
    else:
        print("[INFO] Evaluating FULL TEST dataset")
        test_set = SlideDataset(
            metadata_csv=config["data"]["metadata_csv"],
            organ=config["data"]["organ"],
            split_csv=config["datasets"]["test"],
            features_dir=config["features"]["dir"],
        )

    test_loader = DataLoader(test_set, batch_size=config["train"]["batch_size"])

    # -------------------------------
    # Model setup
    # -------------------------------
    model = SlideClassifier(
        in_dim=config["train"]["in_dim"],
        hidden_dim=config["train"]["hidden_dim"]
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Loss (same as training)
    loss_choice = config["train"]["loss"]
    if loss_choice == "BCE":
        criterion = nn.BCELoss()
    elif loss_choice == "MSE":
        criterion = nn.MSELoss()
    elif loss_choice == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss: {loss_choice}")

    # -------------------------------
    # Evaluation loop
    # -------------------------------
    all_labels, all_preds = [], []
    total_loss = 0

    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(device), labels.to(device)

            preds = model(feats)
            loss = criterion(preds, labels)
            total_loss += loss.item()

            # convert predictions to binary/argmax
            if loss_choice == "CE":
                predicted = torch.argmax(preds, dim=1).cpu().numpy()
                true = labels.cpu().numpy()
            else:
                predicted = (preds > 0.5).int().cpu().numpy()
                true = labels.cpu().numpy()

            all_preds.extend(predicted)
            all_labels.extend(true)

    avg_loss = total_loss / len(test_loader)
    acc, prec, rec, f1 = compute_metrics(all_labels, all_preds)

    print("\n Evaluation Results")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    with open("configs/configs.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_evaluation(config)
