# data/collate.py

def collate_slide(batch):
    """
    batch: List[(Tensor(T, D), label)]
    """
    slides, labels = zip(*batch)
    return list(slides), list(labels)


def collate_animal(batch):
    """
    batch: List[(List[Tensor(T_i, D)], label)]
    """
    animals, labels = zip(*batch)
    return list(animals), list(labels)
