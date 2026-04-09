import torch
import joblib
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


def fit_pca_fast(
    dataset,
    collate_fn,
    device,
    pca_dim=256,
    max_tiles=100000,
    batch_size=16,
    num_workers=4,
    save_path=None
):
    """
    Fit PCA using a random subset of tiles.

    This is much faster than fitting PCA on all tiles.

    Parameters
    ----------
    dataset : Dataset
        Training dataset
    collate_fn : callable
        MIL collate function
    device : str
        cuda or cpu
    pca_dim : int
        Number of PCA components
    max_tiles : int
        Maximum number of tiles used to fit PCA
    batch_size : int
        DataLoader batch size
    num_workers : int
        DataLoader workers
    save_path : Path or str
        Optional path to save PCA model

    Returns
    -------
    pca : IncrementalPCA
    """

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
    )

    sampled_tiles = []
    tile_count = 0

    print("\n[PCA] Sampling tiles for PCA fitting...\n")

    pbar = tqdm(loader, desc="Collecting tiles", ncols=120)

    for X, mask, _ in pbar:

        X = X.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        tiles = X[mask]

        if tiles.numel() == 0:
            continue

        tiles = torch.nn.functional.normalize(tiles, dim=1)

        tiles = tiles.cpu()

        sampled_tiles.append(tiles)

        tile_count += tiles.shape[0]

        if tile_count >= max_tiles:
            break

    sampled_tiles = torch.cat(sampled_tiles)[:max_tiles]

    print(f"\n[PCA] Fitting PCA using {sampled_tiles.shape[0]} tiles\n")

    pca = IncrementalPCA(n_components=pca_dim, batch_size=4096)

    pca.fit(sampled_tiles.numpy())

    print("[PCA] PCA fitted successfully\n")

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(pca, save_path)

        print(f"[PCA] Saved PCA model to {save_path}\n")

    return pca


def load_pca(path):
    """
    Load saved PCA model.
    """
    return joblib.load(path)


def apply_pca(tiles, pca, device):
    """
    Apply PCA transform to tile embeddings.
    """

    tiles_np = tiles.cpu().numpy()

    tiles_pca = pca.transform(tiles_np)

    return torch.from_numpy(tiles_pca).to(device)