import pandas as pd
from pathlib import Path
import yaml

def load_yaml_config(path):
    """Load configuration file to the corresponding path where the config file is located."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def export_wsi_paths(cfg_path):
    cfg= load_yaml_config(cfg_path)
    datasets=cfg["datasets"]
    for split in ["train", "val" , "test"]:
        csv_path = Path(datasets[split])
        if not csv_path.exists():
            print(f"Missing {split} file at {csv_path.name} ")
            continue
        df = pd.read_csv(csv_path)
        if "wsi_path" not in df.columns:
            print(f" No 'wsi_path' column found in {csv_path.name}, skipping.")
            continue
        out_path = csv_path.with_name(csv_path.stem + "_wsi.csv")
        df[["wsi_path"]].dropna().to_csv(out_path, index=False)
        print(f" Saved {len(df)} WSI paths to {out_path}")

if __name__ == "__main__":
    cfg_path = "/data/temporary/mika/repos/oaks_project/pipeline/configs/configs.yaml"
    export_wsi_paths(cfg_path)