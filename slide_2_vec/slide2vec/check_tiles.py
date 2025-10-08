import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# === Adapt this path to your output directory ===
OUTPUT_DIR = Path("/home/mikaklepper/temporary/outputs_slide_2_vec/uni_output")
COORD_DIR = OUTPUT_DIR / "coordinates"
VISU_DIR = OUTPUT_DIR / "visualization" / "tiling"

def main():
    if not COORD_DIR.exists():
        print(f"❌ Coordinates folder not found: {COORD_DIR}")
        return

    coord_files = list(COORD_DIR.glob("*.npy"))
    if not coord_files:
        print("❌ No coordinate files found.")
        return

    print(f"✅ Found {len(coord_files)} coordinate files.")

    # Just check the first slide
    example_file = coord_files[0]
    coords = np.load(example_file)
    print(f"Loaded {example_file.name} with shape {coords.shape}")

    # Print first few coordinates
    print("First 10 coordinates:")
    print(coords[:10])

    # Plot coordinates (just a scatter plot of x,y)
    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.5)
    plt.gca().invert_yaxis()  # match slide coordinates
    plt.title(f"Tile coordinates: {example_file.stem}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    # Also check visualization images
    if VISU_DIR.exists():
        visu_files = list(VISU_DIR.glob("*.jpg"))
        print(f"✅ Found {len(visu_files)} visualization images.")
        if visu_files:
            print(f"Example visualization: {visu_files[0]}")
    else:
        print("⚠️ No visualization directory found.")

if __name__ == "__main__":
    main()

