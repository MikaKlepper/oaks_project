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
from sklearn.svm import LinearSVC, SVC



class BaseProbe:
    def fit(self, dataset):
        raise NotImplementedError

    def predict(self, dataset):
        raise NotImplementedError

    def predict_proba(self, dataset):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError


class LinearHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Torch Probe Config
# ============================================================
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
    rel_tolerance: float = 0.01  # relative tolerance for early stopping

    def make_loss(self):
        name = self.loss.lower()
        if name == "crossentropy":
            return nn.CrossEntropyLoss()
        raise ValueError(f"Unsupported loss: {self.loss}")

    def make_optimizer(self, params):
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


# ============================================================
# TorchProbe
# ============================================================
# ============================================================
# TorchProbe (FAST VERSION)
# ============================================================
class TorchProbe(BaseProbe):
    def __init__(self, model: nn.Module, cfg: TorchProbeConfig):
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
                return step / max(1, warmup_steps)     # linear warmup
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def fit(self, dataset):
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        steps_per_epoch = len(loader)
        scheduler = self._build_scheduler(steps_per_epoch)

        logging.info(f"[TorchProbe] Training on device: {self.cfg.device}")
        logging.info(str(self.model))

        best_loss = float("inf")
        best_state = None
        global_step = 0

        # early stopping
        patience = self.cfg.patience
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

            for feats, labels in pbar:
                feats = feats.to(self.cfg.device, non_blocking=True)
                labels = labels.to(self.cfg.device, non_blocking=True)

                self.optimizer.zero_grad()
                logits = self.model(feats)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                scheduler.step()

                global_step += 1
                running_loss += loss.item() * feats.size(0)

            # compute epoch avg
            epoch_loss = running_loss / len(dataset)
            lr = scheduler.get_last_lr()[0]

            tqdm.write(
                f"[TorchProbe] Epoch {epoch+1}/{self.cfg.epochs} | "
                f"Loss = {epoch_loss:.4f} | LR = {lr:.2e}"
            )
            # early stopping check
            improvement = best_loss - epoch_loss
            required_improvement = best_loss * self.cfg.rel_tolerance
            if best_loss == float("inf") or improvement > required_improvement:
                # Accept improvement or first epoch
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                # No significant improvement
                if epoch - best_epoch >= patience:
                    tqdm.write(
                        f"[TorchProbe] Early stopping at epoch {epoch+1} "
                        f"(no improvement for {patience} epochs)"
                    )
                    break

        if best_state:
            self.model.load_state_dict(best_state)
            logging.info(f"[TorchProbe] Restored best model (loss={best_loss:.4f})")

    def predict(self, dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), num_workers=4)
        preds = []

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(
                loader,
                desc="Predict",
                ncols=120,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
            )
            for feats, _ in pbar:
                feats = feats.to(self.cfg.device)
                logits = self.model(feats)
                preds.append(torch.argmax(logits, dim=1).cpu())

        return torch.cat(preds).numpy()

    def predict_proba(self, dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), num_workers=4)
        probs = []

        self.model.eval()
        with torch.no_grad():
            pbar = tqdm(
                loader,
                desc="Predict Proba",
                ncols=120,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
            )
            for feats, _ in pbar:
                feats = feats.to(self.cfg.device)
                logits = self.model(feats)
                probs.append(torch.softmax(logits, dim=1).cpu())

        return torch.cat(probs).numpy()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f"[TorchProbe] Saved checkpoint to {path}")

    def load(self, path):
        path = Path(path)
        state = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(state)
        logging.info(f"[TorchProbe] Loaded checkpoint from {path}")


# ============================================================
# Sklearn Probe
# ============================================================
class SklearnProbe(BaseProbe):
    def __init__(self, model):
        self.model = model

    def fit(self, dataset):
        loader = DataLoader(dataset, batch_size=1024, num_workers=4)
        X_list, y_list = [], []

        pbar = tqdm(
            loader,
            desc="Sklearn: load data",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]",
        )
        for feats, labels in pbar:
            X_list.append(feats.numpy())
            y_list.append(labels.numpy())

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        logging.info(f"[SklearnProbe] Fitting {self.model.__class__.__name__}")
        self.model.fit(X, y)

    def predict(self, dataset):
        loader = DataLoader(dataset, batch_size=1024, num_workers=2)
        preds = []

        pbar = tqdm(
            loader,
            desc="Sklearn Predict",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]",
        )
        for feats, _ in pbar:
            preds.append(self.model.predict(feats.numpy()))

        return np.concatenate(preds)

    def predict_proba(self, dataset):
        if not hasattr(self.model, "predict_proba"):
            raise NotImplementedError("This sklearn model has no predict_proba()")

        loader = DataLoader(dataset, batch_size=1024, num_workers=2)
        probs = []

        pbar = tqdm(
            loader,
            desc="Sklearn Predict Proba",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]",
        )
        for feats, _ in pbar:
            probs.append(self.model.predict_proba(feats.numpy()))

        return np.concatenate(probs, axis=0)

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
    """
    Always save/load the probe checkpoint inside the TRAIN folder of the experiment:
        <experiment_root>/train/probe_<type>.pt or .joblib
    """

    suffix = ".pt" if is_torch else ".joblib"
    probe_type = prepared["probe"]["type"]

    ckpt_dir = Path(exp_root) / "train"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    filename = f"probe_{probe_type}{suffix}"
    return ckpt_dir / filename

# ============================================================
# Probe Factory
# ============================================================
def build_probe(prepared, input_dim: int, num_classes: int) -> BaseProbe:
    p = prepared["probe"]
    r = prepared["runtime"]

    device = r.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Torch probes
    if t == "linear":
        return TorchProbe(LinearHead(input_dim, num_classes), cfg)
    if t == "mlp":
        return TorchProbe(MLPClassifier(input_dim, cfg.hidden_dim, num_classes, cfg.num_layers), cfg)

    # Sklearn probes
    if t == "logreg":
        return SklearnProbe(LogisticRegression(max_iter=1000, n_jobs=-1))
    if t == "knn":
        return SklearnProbe(KNeighborsClassifier(n_neighbors=p["knn_neighbors"], n_jobs=-1))
    # if t == "svm_linear":
    #     return SklearnProbe(LinearSVC())
    if t == "svm_linear":
        return SklearnProbe(SVC(kernel="linear", probability=True))
    if t == "svm_rbf":
        return SklearnProbe(SVC(kernel="rbf", probability=True))

    raise ValueError(f"Unknown probe type: {p['type']}")
