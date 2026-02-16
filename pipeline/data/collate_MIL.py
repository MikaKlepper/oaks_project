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

    lengths = [b.size(0) for b in bags]
    max_len = max(lengths)
    D = bags[0].size(1)

    X = torch.zeros(len(bags), max_len, D)
    mask = torch.zeros(len(bags), max_len, dtype=torch.bool)

    for i, bag in enumerate(bags):
        n = bag.size(0)
        X[i, :n] = bag
        mask[i, :n] = True

    y = torch.tensor(labels, dtype=torch.long)

    return X, mask, y
