# data/collate_MIL.py

import torch

def collate_mil(batch):
    """
    Collate function for Multiple Instance Learning (MIL).

    Pads variable-length bags of tile embeddings to the maximum
    bag length in the batch.

    Parameters
    ----------
    batch : list of tuples
        Each element is (bag, label), where:
            - bag   : Tensor of shape (Ni, D)
            - label : int

    Returns
    -------
    X : torch.Tensor
        Padded bag tensor of shape (B, N_max, D)

    mask : torch.BoolTensor
        Boolean mask of shape (B, N_max)
        True indicates a valid tile, False indicates padding

    y : torch.LongTensor
        Labels of shape (B,)
    """
    bags, labels = zip(*batch)
    batch_size = len(bags)
    max_len = max(bag.size(0) for bag in bags)
    emb_dim = bags[0].size(1)

    X = torch.zeros(batch_size, max_len, emb_dim, dtype=bags[0].dtype)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, bag in enumerate(bags):
        length = bag.size(0)
        X[i, :length] = bag
        mask[i, :length] = True

    return X, mask, torch.tensor(labels, dtype=torch.long)
