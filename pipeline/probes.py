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

from sklearn.decomposition import IncrementalPCA

from torchmil.models import ABMIL
from torchmil.models import CLAM_SB as CLAM
from torchmil.models import DSMIL
import os
import random
import normflows as nf



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loader(
    dataset,
    *,
    batch_size: int,
    collate_fn=None,
    shuffle: bool = False,
    num_workers: int = 0,
    generator=None,
    worker_init_fn=None,
    prefetch_factor: int | None = None,
):
    use_workers = num_workers > 0
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=use_workers,
        persistent_workers=use_workers,
        generator=generator,
        worker_init_fn=worker_init_fn if use_workers else None,
    )
    if use_workers and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


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
    patience: int = 3
    rel_tolerance: float = 0.01
    max_mil_epochs = 15
    min_delta: float = 2e-3
    seed: int = 42

    # flow-specific params
    flow_train_max_tiles: int = 20000
    flow_topk_frac: float = 0.05
    flow_tau_percentile: float = 95
    flow_pca_fit_max_tiles: int = 200000
    flow_input_dim: int = 64


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
        set_seed(cfg.seed)
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)

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

    def _make_loader(
        self,
        dataset,
        *,
        collate_fn=None,
        shuffle: bool = False,
        num_workers: int | None = None,
        prefetch_factor: int | None = None,
    ):
        if num_workers is None:
            num_workers = self.cfg.num_workers
        return make_loader(
            dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=self.generator,
            worker_init_fn=seed_worker,
            prefetch_factor=prefetch_factor,
        )

    def fit(self, dataset, collate_fn=None):
        """
        Train the probe on the given dataset.
        """
        loader = self._make_loader(dataset, collate_fn=collate_fn, shuffle=True)

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
                self.optimizer.zero_grad(set_to_none=True)
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
        loader = self._make_loader(dataset, collate_fn=collate_fn, shuffle=False)

        self.model.eval()
        preds = []

        with torch.inference_mode():
            pbar = tqdm(loader, desc="Predict", ncols=120)
            for batch in pbar:
                logits, _ = self.forward_batch(batch)
                preds.append(torch.argmax(logits, dim=1).cpu())

        return torch.cat(preds).numpy()

    def predict_proba(self, dataset, collate_fn=None):
        """
        Predict class probabilities for the given dataset.
        """
        loader = self._make_loader(dataset, collate_fn=collate_fn, shuffle=False)

        self.model.eval()
        probs = []

        with torch.inference_mode():
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

def build_flow(
    input_dim: int,
    num_layers: int = 8,         
    hidden_dim: int = 256,         
):

    flows = []

    for _ in range(num_layers):

        net = nf.nets.MLP(
            [
                input_dim // 2,
                hidden_dim,
                hidden_dim,
                hidden_dim,
                hidden_dim,
                input_dim,
            ],
            init_zeros=True,
        )
        flows.append(nf.flows.ActNorm(input_dim))

        flows.append(
            nf.flows.AffineCouplingBlock(
                param_map=net
            )
        )

        flows.append(
            nf.flows.Permute(input_dim, mode="shuffle")
        )

    base = nf.distributions.base.DiagGaussian(
        input_dim,
        trainable=True
    )

    return nf.NormalizingFlow(base, flows)


class TorchPCA(nn.Module):
    def __init__(self, components, mean):
        super().__init__()
        self.register_buffer("components", components)
        self.register_buffer("mean", mean)

    def forward(self, x):
        return (x - self.mean) @ self.components.T

class FlowProbe(TorchProbe):

    def __init__(self, model, cfg: TorchProbeConfig, pca=None):
        set_seed(cfg.seed)

        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.pca = pca

        self.projector = None

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.epochs,
        )

        self.generator = torch.Generator().manual_seed(cfg.seed)

        self._pred_cache = None
        self.tau = None

        self.train_max_tiles = cfg.flow_train_max_tiles
        self.topk_frac = cfg.flow_topk_frac
        self.tau_percentile = cfg.flow_tau_percentile
        self.pca_fit_max_tiles = cfg.flow_pca_fit_max_tiles

        self.flow_input_dim = cfg.flow_input_dim

        self.device = cfg.device
        self.use_amp = "cuda" in cfg.device

        print(f"[FlowProbe] Using flow_input_dim = {self.flow_input_dim}")

    # -------------------------
    # PCA
    # -------------------------
    def _build_projector(self):
        if self.pca is None:
            self.projector = None
            return

        components = torch.tensor(
            self.pca.components_, dtype=torch.float32, device=self.device
        )
        mean = torch.tensor(
            self.pca.mean_, dtype=torch.float32, device=self.device
        )

        self.projector = TorchPCA(components, mean).to(self.device)

        # remove sklearn dependency
        self.pca = None

    def _apply_pca(self, tiles):
        if self.projector is None:
            return tiles
        return self.projector(tiles)

    # -------------------------
    # HELPERS
    # -------------------------
    def _project(self, tiles):
        tiles = self._apply_pca(tiles)

        if tiles.shape[1] != self.flow_input_dim:
            raise ValueError(
                f"[FlowProbe] Dim mismatch: got {tiles.shape[1]}, expected {self.flow_input_dim}"
            )

        # return torch.nn.functional.normalize(tiles, dim=1)
        return tiles

    def _extract_tiles(self, X, mask):
        B, N, D = X.shape
        tiles = X.reshape(B * N, D)
        mask = mask.reshape(B * N).bool()
        return tiles[mask]

    def _subsample(self, tiles):
        if tiles.shape[0] <= self.train_max_tiles:
            return tiles
        idx = torch.randperm(tiles.shape[0])[:self.train_max_tiles]
        return tiles[idx]

    def _score_tiles(self, tiles):
        return -self.model.log_prob(tiles)

    def _aggregate_slide(self, X, mask):
        B, N, D = X.shape

        tiles = X.reshape(B * N, D)
        mask_flat = mask.reshape(B * N).bool()
        tiles = tiles[mask_flat]

        if tiles.numel() == 0:
            return torch.zeros(B, device=X.device)

        tiles = self._project(tiles)
        scores = self._score_tiles(tiles)

        slide_ids = torch.arange(B, device=X.device).repeat_interleave(N)
        slide_ids = slide_ids[mask_flat]
        order = torch.argsort(slide_ids)
        sorted_ids = slide_ids[order]
        sorted_scores = scores[order]
        counts = torch.bincount(sorted_ids, minlength=B)

        slide_scores = []
        start = 0
        for b in range(B):
            c = int(counts[b].item())
            if c == 0:
                slide_scores.append(torch.tensor(0.0, device=X.device))
                continue
            s = sorted_scores[start:start + c]
            start += c
            k = max(1, int(self.topk_frac * c))
            topk_vals = s.topk(k).values
            slide_scores.append(torch.quantile(topk_vals, 0.8))

        return torch.stack(slide_scores)

    # -------------------------
    # PCA FIT (FAST VERSION)
    # -------------------------
    def _fit_pca(self, dataset, collate_fn=None):
        if self.pca is None:
            return

        print("[FlowProbe] Fitting PCA...")

        loader = self._make_loader(
            dataset,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=0,
        )

        collected = 0

        for X, mask, _ in tqdm(loader, desc="PCA Fit", ncols=120):

            B, N, D = X.shape

            tiles = X.reshape(B * N, D)
            mask = mask.reshape(B * N).bool()
            tiles = tiles[mask]

            if tiles.numel() == 0:
                continue

            remaining = self.pca_fit_max_tiles - collected
            if remaining <= 0:
                break

            if tiles.shape[0] > remaining:
                idx = torch.randperm(tiles.shape[0])[:remaining]
                tiles = tiles[idx]

            self.pca.partial_fit(tiles.numpy())
            collected += tiles.shape[0]

        print(f"[FlowProbe] PCA fitted on {collected} tiles")

        self._build_projector()

    # -------------------------
    # TRAIN
    # -------------------------
    def fit(self, dataset, collate_fn=None):

        self._pred_cache = None

        if self.pca is not None:
            self._fit_pca(dataset, collate_fn)

        loader = self._make_loader(dataset, collate_fn=collate_fn, shuffle=True)

        scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        best_loss = float("inf")
        best_state = None
        best_epoch = 0

        for epoch in range(self.cfg.epochs):

            self.model.train()
            running_loss = 0
            total_tiles = 0

            pbar = tqdm(loader, desc=f"Epoch {epoch+1}", ncols=120)

            for X, mask, _ in pbar:

                # CPU first
                tiles = self._extract_tiles(X, mask)
                if tiles.numel() == 0:
                    continue

                # subsample BEFORE GPU
                tiles = self._subsample(tiles)

                # move to GPU
                tiles = tiles.to(self.device, non_blocking=True)

                tiles = self._project(tiles)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=self.use_amp):
                    loss = self.model.forward_kld(tiles)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                running_loss += loss.item() * tiles.shape[0]
                total_tiles += tiles.shape[0]

                pbar.set_postfix(loss=f"{loss.item():.2f}")

            self.scheduler.step()

            epoch_loss = running_loss / max(total_tiles, 1)
            tqdm.write(f"[FlowProbe] Epoch {epoch+1} | Loss={epoch_loss:.4f}")

            if best_loss == float("inf") or (best_loss - epoch_loss > self.cfg.min_delta):
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            elif epoch - best_epoch >= self.cfg.patience:
                break

        if best_state:
            self.model.load_state_dict(best_state)

        scores = self._predict_scores_cached(dataset, collate_fn)
        self.tau = float(np.percentile(scores, self.tau_percentile))

        print(f"[FlowProbe] tau = {self.tau:.4f}")

    # -------------------------
    # INFERENCE
    # -------------------------
    def _predict_scores_cached(self, dataset, collate_fn=None):

        key = (id(dataset), len(dataset), self.cfg.batch_size)

        if self._pred_cache is not None and self._pred_cache["key"] == key:
            return self._pred_cache["scores"]

        loader = self._make_loader(dataset, collate_fn=collate_fn, shuffle=False)

        self.model.eval()
        all_scores = []

        with torch.inference_mode():
            for X, mask, _ in tqdm(loader, desc="Inference", ncols=120):
                X = X.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                slide_scores = self._aggregate_slide(X, mask)
                all_scores.append(slide_scores.cpu())

        scores = torch.cat(all_scores).numpy()

        self._pred_cache = {"key": key, "scores": scores}
        return scores

    # -------------------------
    # PREDICT
    # -------------------------
    def predict(self, dataset, collate_fn=None):
        if self.tau is None:
            raise RuntimeError("tau is None → model not trained or loaded")

        scores = self._predict_scores_cached(dataset, collate_fn)
        return (scores > self.tau).astype(np.int64)

    def predict_proba(self, dataset, collate_fn=None):
        if self.tau is None:
            raise RuntimeError("tau is None → model not trained or loaded")

        scores = self._predict_scores_cached(dataset, collate_fn)

        x = np.clip(scores - self.tau, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    # -------------------------
    # SAVE / LOAD
    # -------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        pca_state = None
        if self.projector is not None:
            pca_state = {
                "components": self.projector.components.cpu(),
                "mean": self.projector.mean.cpu(),
            }

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "tau": self.tau,
                "pca": pca_state,
            },
            path,
        )

        print(f"[FlowProbe] Saved to {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

        self.tau = ckpt["tau"]

        pca = ckpt["pca"]
        if pca is not None:
            self.projector = TorchPCA(
                pca["components"].to(self.device),
                pca["mean"].to(self.device),
            )
        else:
            self.projector = None

        self._pred_cache = None

        print(f"[FlowProbe] Loaded from {path}")

        
class MILTorchProbe(TorchProbe):
    """
    MIL probe: uses torchmil compute_loss and predict.
    """

    def fit(self, dataset, collate_fn=None):
        """
        Train the MIL probe on the given dataset.
        """
        if len(dataset) > 1000:
            num_workers = self.cfg.num_workers
        else:
            num_workers = 0

        loader = self._make_loader(
            dataset,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        scheduler = self._build_scheduler(len(loader))
        best_loss = float("inf")
        best_state = None
        best_epoch = 0

        max_epochs = min(self.cfg.epochs, self.cfg.max_mil_epochs)
        for epoch in range (max_epochs):
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

                self.optimizer.zero_grad(set_to_none=True)

                _, loss_dict = self.model.compute_loss(labels, X, mask)
                loss = loss_dict["BCEWithLogitsLoss"]
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
            improvement = best_loss - epoch_loss
            # if best_loss == float("inf") or epoch_loss < best_loss * (1 - self.cfg.rel_tolerance):
            if best_loss == float("inf") or improvement > self.cfg.min_delta:
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            elif epoch - best_epoch >= self.cfg.patience:
                tqdm.write(f"[MILTorchProbe] Early stopping at epoch {epoch+1}")
                break

        if best_state:
            self.model.load_state_dict(best_state)
            logging.info(f"[MILTorchProbe] Restored best model (loss={best_loss:.4f})")

    # chache logits to avoid running MIL inference twice during predict and predict_proba

    def _cache_key(self, dataset, collate_fn):
        # Cache is valid only for the same dataset object + collate_fn + batch_size.
        return (id(dataset), id(collate_fn), len(dataset), self.cfg.batch_size)

    def _predict_logits_cached(self, dataset, collate_fn=None) -> np.ndarray:
        """
        Run MIL inference once and cache slide-level logits on CPU.
        Returns logits as a 1D numpy array of shape (N,).
        """
        key = self._cache_key(dataset, collate_fn)

        if getattr(self, "_pred_cache", None) is not None and self._pred_cache.get("key") == key:
            return self._pred_cache["logits"]

        num_workers = self.cfg.num_workers
        loader = self._make_loader(
            dataset,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        self.model.eval()
        logits_list = []

        with torch.inference_mode():
            for X, mask, _ in loader:
                X = X.to(self.cfg.device, non_blocking=True)
                mask = mask.to(self.cfg.device, non_blocking=True)

                logits = self.model.predict(X, mask, return_inst_pred=False)
                logits_list.append(logits.detach().cpu())

        logits = torch.cat(logits_list)

        # ensure shape (N,) no matter if logits are (N,) or (N,1)
        logits_np = logits.numpy().reshape(-1)

        # cache only logits (tiny memory)
        self._pred_cache = {"key": key, "logits": logits_np}

        return logits_np

    def predict(self, dataset, collate_fn=None):
        """
        Predict hard labels (0/1) for the given dataset.
        Uses cached logits to avoid running MIL inference twice.
        """
        logits = self._predict_logits_cached(dataset, collate_fn)

        # sigmoid(logits) >= 0.5  <=>  logits >= 0
        preds = (logits >= 0.0).astype(np.int64)
        return preds

    def predict_proba(self, dataset, collate_fn=None):
        """
        Predict probabilities for the positive class for the given dataset.
        Uses cached logits to avoid running MIL inference twice.
        """
        logits = self._predict_logits_cached(dataset, collate_fn)

        # sigmoid in numpy
        probs = 1.0 / (1.0 + np.exp(-logits))
        return probs


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
    """
    Return the default path for a probe model checkpoint.

    Parameters
    ----------
    prepared : dict
        Parsed config containing probe and runtime information.
    exp_root : str
        Root directory of the experiment.
    is_torch : bool, optional
        Whether the probe model is a PyTorch model (default: True).

    Returns
    -------
    Path
        Default path for the probe model checkpoint.
    """
    suffix = ".pt" if is_torch else ".joblib"
    probe_type = prepared["probe"]["type"].lower()
    ckpt_dir = Path(exp_root) / "train"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / f"probe_{probe_type}{suffix}"


def build_probe(prepared, input_dim: int, num_classes: int):
    """
    Build a probe model based on the prepared config.

    Args:
        prepared (dict): Parsed config containing probe and runtime information.
        input_dim (int): Number of input features.
        num_classes (int): Number of output classes.

    Returns:
        TorchProbe or SklearnProbe: A probe model with the specified architecture and config.
    """
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
        seed=r.get("seed", 42),

        # flow params
        flow_input_dim=p.get("flow_input_dim", input_dim),
        flow_train_max_tiles=p.get("flow_train_max_tiles", 20000),
        flow_topk_frac=p.get("flow_topk_frac", 0.05),
        flow_tau_percentile=p.get("flow_tau_percentile", 95),
        flow_pca_fit_max_tiles=p.get("flow_pca_fit_max_tiles", 200000),
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
   
    if t == "flow":

        print(f"[Flow] input_dim = {input_dim}")
        print(f"[Flow] flow_input_dim = {cfg.flow_input_dim}")

        pca = None
        if cfg.flow_input_dim < input_dim:
            print("[Flow] PCA ENABLED")
            pca = IncrementalPCA(n_components=cfg.flow_input_dim)
        else:
            print("[Flow] PCA DISABLED")

        model = build_flow(
            input_dim=cfg.flow_input_dim,
            hidden_dim=p.get("flow_hidden", 256),   # still fine here
            num_layers=p.get("flow_layers", 8),
        )

        return FlowProbe(model, cfg, pca=pca)

    if t == "logreg":
        return SklearnProbe(LogisticRegression(max_iter=1000, random_state=cfg.seed))

    if t == "knn":
        return SklearnProbe(KNeighborsClassifier(n_neighbors=p["knn_neighbors"]))

    if t == "svm_linear":
        return SklearnProbe(SVC(kernel="linear", probability=True, random_state=cfg.seed))

    if t == "svm_rbf":
        return SklearnProbe(SVC(kernel="rbf", probability=True, random_state=cfg.seed))

    raise ValueError(f"Unknown probe type: {p['type']}")
