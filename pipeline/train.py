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
