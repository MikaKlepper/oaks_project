from argparser import get_args
from utils.config_loader import load_merged_config
from data.create_datasets import AnimalDataset
import pprint

def main():
    args = get_args()
    cfg = load_merged_config(args.config, args)

    print("\n===== FINAL CONFIG =====")
    pprint.pp(cfg)

    

if __name__ == "__main__":
    main()
    
