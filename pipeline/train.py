# train.py
# Train a simple classifier on slide-level features for hypertrophy prediction.

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data.create_datasets import SlideDataset   # <- your dataset class
from wsi_models.wsi_models import SlideClassifier  # <- your model class
from pathlib import Path
import yaml


def run_training(config):
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data set choice, subset or full split
    if config["datasets"]["use_subset"]:
        print("[INFO] Using SUBSET dataset")
        train_set = SlideDataset(
            metadata_csv=config["data"]["metadata_csv"],
            organ=config["data"]["organ"],
            subset_csv=config["datasets"]["subset"],
            features_dir=config["features"]["dir"],
            subset_fraction=config["datasets"].get("subset_fraction", None),
        )
        val_set = None  # no val in subset mode
    else:
        print("[INFO] Using FULL SPLIT datasets")
        train_set = SlideDataset(
            metadata_csv=config["data"]["metadata_csv"],
            organ=config["data"]["organ"],
            split_csv=config["datasets"]["train"],
            features_dir=config["features"]["dir"],
            subset_fraction=config["datasets"].get("subset_fraction", None),
        )
        val_set = SlideDataset(
            metadata_csv=config["data"]["metadata_csv"],
            organ=config["data"]["organ"],
            split_csv=config["datasets"]["val"],
            features_dir=config["features"]["dir"],
        )

    print(f"[INFO] Train set size: {len(train_set)}")
    if val_set:
        print(f"[INFO] Val set size: {len(val_set)}")
    else:
        print("[INFO] No validation set (subset mode)")

    # Data loaders   
    train_loader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["train"]["batch_size"]) if val_set else None

    # define model, loss, optimizer
    model = SlideClassifier(
        in_dim=config["train"]["in_dim"],
        hidden_dim=config["train"]["hidden_dim"]
    ).to(device)

    # loss configuration check
    loss_choice = config["train"]["loss"]
    if loss_choice == "BCE":
        criterion = nn.BCELoss()
    elif loss_choice == "MSE":
        criterion = nn.MSELoss()
    elif loss_choice == "CE":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss: {loss_choice}")
    
    # optimizer configuration check
    opt_choice = config["train"]["optimizer"]
    if opt_choice == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["train"]["lr"])
    elif opt_choice == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["train"]["lr"], momentum=0.9)
    elif opt_choice == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config["train"]["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {opt_choice}")

    # full training loop
    for epoch in range(config["train"]["epochs"]):
        model.train()
        total_loss = 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            preds = model(feats)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['train']['epochs']} | Train Loss: {avg_train_loss:.4f}")

        # validation step only in full mode
        if val_loader:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats, labels = feats.to(device), labels.to(device)
                    preds = model(feats)
                    loss = criterion(preds, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f" Validation Loss: {avg_val_loss:.4f}")

    # save the model
    Path("outputs").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "outputs/slide_classifier.pth")
    print("Training complete. Model saved to outputs/slide_classifier.pth")

# main entry
if __name__ == "__main__":
    with open("configs/configs.yaml", "r") as f:
        config = yaml.safe_load(f)
    run_training(config)




# pipeline/train.py

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from argparser import get_args
from utils.config_loader import load_merged_config
from data.prepare_dataset import prepare_dataset_inputs
from data.features_per_animal import group_features_by_animal
from data.dataset_check import check_subset_consistency
from data.create_datasets import ToxicologyDataset
from probes import build_probe, TorchProbe, SklearnProbe, default_probe_path
from logger import setup_logger


def _dataset_to_numpy(dataset, batch_size: int = 256):
    """
    Convert ToxicologyDataset -> (X, y) numpy arrays.
    We only load tensors from .pt here, no gradient tracking.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feats_list = []
    labels_list = []

    for feats, labels in loader:
        feats_list.append(feats)
        labels_list.append(labels)

    X = torch.cat(feats_list, dim=0).cpu().numpy()
    y = torch.cat(labels_list, dim=0).cpu().numpy()

    return X, y


def run_train(cfg):
    exp_root = Path(cfg.experiment_root)
    setup_logger(exp_root)
    logging.info("========== TRAIN STAGE ==========")

    # 1) Prepare dataset inputs
    prepared = prepare_dataset_inputs(cfg)

    # 2) Build animal features if needed
    if prepared["features_type"] == "animal":
        logging.info("[Train] Building animal-level features...")
        group_features_by_animal(prepared)

    # 3) Consistency checks
    check_subset_consistency(prepared)

    # 4) Build dataset
    ds = ToxicologyDataset(prepared)
    logging.info(f"[Train] Dataset size: {len(ds)}")

    # 5) Convert to numpy for probes
    X, y = _dataset_to_numpy(ds, batch_size=cfg.runtime.batch_size)
    input_dim = X.shape[1]
    num_classes = int(y.max() + 1)

    logging.info(f"[Train] Input dim: {input_dim}, Num classes: {num_classes}")

    # 6) Build & train probe
    probe = build_probe(cfg, input_dim, num_classes)
    probe.fit(X, y)

    # 7) Save checkpoint
    is_torch = isinstance(probe, TorchProbe)
    ckpt_path = default_probe_path(cfg, exp_root, is_torch=is_torch)
    probe.save(ckpt_path)

    logging.info("========== TRAIN STAGE DONE ==========")
    return probe, ckpt_path


if __name__ == "__main__":
    args = get_args()
    cfg, _ = load_merged_config(args.config, args)
    run_train(cfg)
