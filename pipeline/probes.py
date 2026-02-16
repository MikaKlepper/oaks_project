# pipeline/probes.py
import logging
import math
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import joblib
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from torchmil.models import ABMIL
from torchmil.models import CLAM_SB as CLAM
from torchmil.models import DSMIL


class BaseProbe:
    def fit(self, dataset, collate_fn=None):
        raise NotImplementedError

    def predict(self, dataset, collate_fn=None):
        raise NotImplementedError

    def predict_proba(self, dataset, collate_fn=None):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class LinearHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        """
        Initialize a LinearHead model.
        """
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        """
        Initialize an MLPClassifier model.
        """
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class TorchProbeConfig:
    probe_type: str
    hidden_dim: int
    num_layers: int

    epochs: int
    lr: float
    batch_size: int
    device: str

    optimizer: str
    weight_decay: float
    momentum: float
    loss: str
    num_workers: int = 4
    patience: int = 10
    rel_tolerance: float = 0.01

    def make_loss(self):
        """
        Create a loss function based on the config.
        """
        name = self.loss.lower()
        if name == "crossentropy":
            return nn.CrossEntropyLoss()
        if name == "bce":
            return nn.BCEWithLogitsLoss()
        raise ValueError(f"Unsupported loss: {self.loss}")

    def make_optimizer(self, params):
        """
        Create an optimizer based on the config.
        """
        name = self.optimizer.lower()
        if name == "adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "adamw":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "sgd":
            return optim.SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {self.optimizer}")


class TorchProbe(BaseProbe):
    def __init__(self, model: nn.Module, cfg: TorchProbeConfig):
        """
        Initialize a TorchProbe instance.
        """
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.criterion = cfg.make_loss()
        self.optimizer = cfg.make_optimizer(self.model.parameters())

    def _build_scheduler(self, steps_per_epoch: int):
        """Warmup + Cosine LR schedule (per batch)."""
        total_steps = self.cfg.epochs * steps_per_epoch
        warmup_steps = int(0.1 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def forward_batch(self, batch):
        """
        Forward a single batch through the model.
        """
        feats, labels = batch
        feats = feats.to(self.cfg.device, non_blocking=True)
        labels = labels.to(self.cfg.device, non_blocking=True)
        logits = self.model(feats)
        return logits, labels

    def fit(self, dataset, collate_fn=None):
        """
        Train the probe on the given dataset.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        scheduler = self._build_scheduler(len(loader))
        best_loss = float("inf")
        best_state = None
        best_epoch = 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{self.cfg.epochs}",
                ncols=120,
                leave=True,
            )

            for batch in pbar:
                self.optimizer.zero_grad()
                logits, labels = self.forward_batch(batch)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                running_loss += loss.item() * labels.size(0)

            epoch_loss = running_loss / len(dataset)
            lr = scheduler.get_last_lr()[0]

            tqdm.write(
                f"[TorchProbe] Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Loss = {epoch_loss:.4f} | LR = {lr:.2e}"
            )

            improvement = best_loss - epoch_loss
            required_improvement = best_loss * self.cfg.rel_tolerance
            if best_loss == float("inf") or improvement > required_improvement:
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                if epoch - best_epoch >= self.cfg.patience:
                    tqdm.write(
                        f"[TorchProbe] Early stopping at epoch {epoch+1}"
                    )
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            logging.info(f"[TorchProbe] Restored best model (loss={best_loss:.4f})")

    def predict(self, dataset, collate_fn=None):
        """
        Predict class labels for the given dataset.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
        )

        self.model.eval()
        preds = []

        with torch.no_grad():
            pbar = tqdm(loader, desc="Predict", ncols=120)
            for batch in pbar:
                logits, _ = self.forward_batch(batch)
                preds.append(torch.argmax(logits, dim=1).cpu())

        return torch.cat(preds).numpy()

    def predict_proba(self, dataset, collate_fn=None):
        """
        Predict class probabilities for the given dataset.
        """
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
        )

        self.model.eval()
        probs = []

        with torch.no_grad():
            pbar = tqdm(loader, desc="Predict Proba", ncols=120)
            for batch in pbar:
                logits, _ = self.forward_batch(batch)

                if logits.ndim == 1:
                    probs.append(torch.sigmoid(logits).cpu())
                else:
                    probs.append(torch.softmax(logits, dim=1).cpu())

        return torch.cat(probs).numpy()

    def save(self, path):
        """
        Save the model checkpoint to the specified path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f"[TorchProbe] Saved checkpoint to {path}")

    def load(self, path):
        """
        Load the model checkpoint from the specified path.
        """
        path = Path(path)
        state = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(state)
        logging.info(f"[TorchProbe] Loaded checkpoint from {path}")


class MILTorchProbe(TorchProbe):
    """
    MIL probe: uses torchmil compute_loss and predict.
    """

    def fit(self, dataset, collate_fn=None):
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=False,
            persistent_workers=False,
        )

        scheduler = self._build_scheduler(len(loader))
        best_loss = float("inf")
        best_state = None
        best_epoch = 0

        for epoch in range(self.cfg.epochs):
            self.model.train()
            running_loss = 0.0

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{self.cfg.epochs}",
                ncols=120,
                leave=True,
            )

            for X, mask, labels in pbar:
                X = X.to(self.cfg.device, non_blocking=True)
                mask = mask.to(self.cfg.device, non_blocking=True)
                labels = labels.to(self.cfg.device, non_blocking=True).float()

                self.optimizer.zero_grad()

                _, loss_dict = self.model.compute_loss(labels, X, mask)
                # print("LOSS_DICT TYPE:", type(loss_dict))
                # print("LOSS_DICT KEYS:", list(loss_dict.keys()))
                # print("LOSS_DICT:", {k: (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()})

                # raise SystemExit("Stop after first batch to inspect loss_dict")

                loss = loss_dict['BCEWithLogitsLoss']
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                running_loss += loss.item() * labels.size(0)

            epoch_loss = running_loss / len(dataset)
            lr = scheduler.get_last_lr()[0]

            tqdm.write(
                f"[MILTorchProbe] Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Loss = {epoch_loss:.4f} | LR = {lr:.2e}"
            )

            if best_loss == float("inf") or epoch_loss < best_loss * (1 - self.cfg.rel_tolerance):
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            elif epoch - best_epoch >= self.cfg.patience:
                tqdm.write(f"[MILTorchProbe] Early stopping at epoch {epoch+1}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
            logging.info(f"[MILTorchProbe] Restored best model (loss={best_loss:.4f})")

    def predict(self, dataset, collate_fn=None):
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
        )

        self.model.eval()
        preds = []

        with torch.no_grad():
            for X, mask, _ in loader:
                X = X.to(self.cfg.device)
                mask = mask.to(self.cfg.device)

                Y_pred = self.model.predict(
                    X, mask, return_inst_pred=False
                )

                preds.append((Y_pred > 0).long().cpu())

        return torch.cat(preds).numpy()

    def predict_proba(self, dataset, collate_fn=None):
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
            num_workers=self.cfg.num_workers,
        )

        self.model.eval()
        probs = []

        with torch.no_grad():
            for X, mask, _ in loader:
                X = X.to(self.cfg.device)
                mask = mask.to(self.cfg.device)

                Y_pred = self.model.predict(
                    X, mask, return_inst_pred=False
                )

                probs.append(torch.sigmoid(Y_pred).cpu())

        return torch.cat(probs).numpy()


class SklearnProbe(BaseProbe):
    def __init__(self, model):
        self.model = model

    def fit(self, dataset, collate_fn=None):
        X, y = zip(*(dataset[i] for i in range(len(dataset))))
        self.model.fit(np.stack(X), np.array(y))

    def predict(self, dataset, collate_fn=None):
        X = [dataset[i][0] for i in range(len(dataset))]
        return self.model.predict(np.stack(X))

    def predict_proba(self, dataset, collate_fn=None):
        X = [dataset[i][0] for i in range(len(dataset))]
        return self.model.predict_proba(np.stack(X))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logging.info(f"[SklearnProbe] Saved sklearn model to {path}")

    def load(self, path):
        path = Path(path)
        self.model = joblib.load(path)
        logging.info(f"[SklearnProbe] Loaded sklearn model from {path}")


def default_probe_path(prepared, exp_root, is_torch=True):
    suffix = ".pt" if is_torch else ".joblib"
    probe_type = prepared["probe"]["type"]
    ckpt_dir = Path(exp_root) / "train"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"probe_{probe_type}{suffix}"


def build_probe(prepared, input_dim: int, num_classes: int):
    p = prepared["probe"]
    r = prepared["runtime"]

    device = r.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    cfg = TorchProbeConfig(
        probe_type=p["type"],
        hidden_dim=p["hidden_dim"],
        num_layers=p["num_layers"],
        epochs=r["epochs"],
        lr=r["lr"],
        batch_size=r["batch_size"],
        device=device,
        optimizer=r["optimizer"],
        weight_decay=r["weight_decay"],
        momentum=r["momentum"],
        loss=r["loss"],
        num_workers=r.get("num_workers", 4),
    )

    t = p["type"].lower()

    if t == "linear":
        return TorchProbe(LinearHead(input_dim, num_classes), cfg)

    if t == "mlp":
        return TorchProbe(
            MLPClassifier(input_dim, cfg.hidden_dim, num_classes, cfg.num_layers), cfg
        )

    if t == "abmil":
        model = ABMIL(in_shape=(input_dim,), att_dim=128)
        return MILTorchProbe(model, cfg)

    if t == "clam":
        model = CLAM(in_shape=(input_dim,))
        return MILTorchProbe(model, cfg)
    
    if t == "dsmil":
        model = DSMIL(in_shape=(input_dim,))
        return MILTorchProbe(model, cfg)

    if t == "logreg":
        return SklearnProbe(LogisticRegression(max_iter=1000))

    if t == "knn":
        return SklearnProbe(KNeighborsClassifier(n_neighbors=p["knn_neighbors"]))

    if t == "svm_linear":
        return SklearnProbe(SVC(kernel="linear", probability=True))

    if t == "svm_rbf":
        return SklearnProbe(SVC(kernel="rbf", probability=True))

    raise ValueError(f"Unknown probe type: {p['type']}")
