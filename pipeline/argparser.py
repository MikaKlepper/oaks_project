import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Pipeline runner")
    parser.add_argument("--config", type=str, default="configs/configs.yaml")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--subset", action="store_true", help="Create subset for training/evaluation")
    return parser.parse_args()
