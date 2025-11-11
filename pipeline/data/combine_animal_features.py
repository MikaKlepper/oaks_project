# import torch
# import pandas as pd
# import os
# import shutil
# import yaml
# from pathlib import Path

# def combine_slide_features_by_animal_organ_UID(config):

#     # loading in all configs
#     master_csv = Path(config["data"]["metadata_csv"])
#     organ = config["data"].get("organ", "Liver")
#     slide_feature_dir = Path(config["features"]["slide_dir"])
#     animal_feature_dir = Path(config["features"]["animal_dir"])

#     # creating the animal dir if it not exists yet
#     animal_feature_dir.mkdir(parents=True, exist_ok=True)
    
#     # printing out for debugging to see if eveyrthing was correct
#     print(f"[INFO] Combining slide features for organ: {organ}")
#     print(f"[INFO] Slide feature dir: {slide_feature_dir}")
#     print(f"[INFO] Output animal feature dir: {animal_feature_dir}")

#     # loading in metadata
#     master_df = pd.read_csv(master_csv, encoding="ISO-8859-1") 
#     master_df = master_df[master_df["ORGAN"].str.lower() == organ.lower()]

#     # group slides by animal and create a dict from it
#     animal_slide_dict = master_df.groupby("subject_organ_UID")["slide_id"].apply(list).to_dict()
#     print(f"[INFO] Found {len(animal_slide_dict)} animals for organ '{organ}'")

#     counter = 0
#     for animal_UID, slide_UIDs in animal_slide_dict.items():
#         animal_feature_path = f"{animal_feature_dir}/{animal_UID}.pt"

#         if os.path.exists(animal_feature_path):
#             # print(f"Animal feature file {animal_feature_path} already exists. Skipping...")
#             counter += 1
#             continue

#         # check if there is only one slide for the animal
#         if len(slide_UIDs) == 1:            

#             # copy to animal feature directory directly
#             slide_feature_path = f"{slide_feature_dir}/{slide_UIDs[0]}.pt"
#             shutil.copy(slide_feature_path, animal_feature_path)

#         else:                    
#             for slide_UID in slide_UIDs:

#                 # Load the slide feature
#                 slide_feature_path = f"{slide_feature_dir}/{slide_UID}.pt"
                
#                 # check if slide features exists
#                 if not os.path.exists(slide_feature_path):
#                     print("***********  WARNING *****************")
#                     print(f"Slide feature file {slide_feature_path} does not exist. Skipping...")
#                     print("**************************************")
#                     continue

#                 # Load the slide features
#                 if torch.cuda.is_available():
#                     slide_features = torch.load(slide_feature_path, map_location=torch.device('cuda'))
#                 else:
#                     slide_features = torch.load(slide_feature_path, map_location=torch.device('cpu'))
                
#                 # If first slide
#                 if slide_UID == slide_UIDs[0]:
#                     animal_feature = slide_features
#                 else:
#                     # If not first slide, concatenate the features
#                     animal_feature = torch.cat((animal_feature, slide_features), dim=0)
#                     print("\n#################################################")
#                     print("Concatenated features for animal UID:", animal_UID)
#                     print("Current animal feature shape:", animal_feature.shape)

#             # Save the animal feature
#             torch.save(animal_feature, animal_feature_path)

#         # give percent completed
#         counter += 1
#         percent_completed = (counter / len(animal_slide_dict)) * 100
#         print(f"Processed {counter}/{len(animal_slide_dict)} animals ({percent_completed:.2f}%)")
        
# if __name__ == "__main__":
#     # load config
#     config_path = "configs/configs.yaml"
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
    
#     # run the feature aggregation
#     combine_slide_features_by_animal_organ_UID(config)


import torch
import pandas as pd
import os
import shutil
import yaml
from pathlib import Path

def combine_slide_features_by_animal_organ_UID(config):

    # === Load config ===
    master_csv = Path(config["data"]["metadata_csv"])
    organ = config["data"].get("organ", "Liver")
    slide_feature_dir = Path(config["features"]["slide_dir"])
    animal_feature_dir = Path(config["features"]["animal_dir"])

    # === Prepare output dir ===
    animal_feature_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Combining slide features for organ: {organ}")
    print(f"[INFO] Slide feature dir: {slide_feature_dir}")
    print(f"[INFO] Output animal feature dir: {animal_feature_dir}")

    # === Load metadata ===
    master_df = pd.read_csv(master_csv, encoding="ISO-8859-1")
    master_df = master_df[master_df["ORGAN"].str.lower() == organ.lower()]

    # === Group slides by animal ===
    animal_slide_dict = master_df.groupby("subject_organ_UID")["slide_id"].apply(list).to_dict()
    print(f"[INFO] Found {len(animal_slide_dict)} animals for organ '{organ}'")

    # === Iterate through animals ===
    counter = 0
    skipped_animals = 0

    for animal_UID, slide_UIDs in animal_slide_dict.items():
        animal_feature_path = animal_feature_dir / f"{animal_UID}.pt"

        # Skip if already exists
        if animal_feature_path.exists():
            counter += 1
            continue

        available_features = []

        # Collect available slide features
        for slide_UID in slide_UIDs:
            slide_feature_path = slide_feature_dir / f"{slide_UID}.pt"
            if not slide_feature_path.exists():
                print(f"[WARN] Missing feature for slide {slide_UID}, skipping...")
                continue
            available_features.append(slide_feature_path)

        # Skip if no features found
        if len(available_features) == 0:
            print(f"[WARN] No available features for animal {animal_UID}, skipping this animal.")
            skipped_animals += 1
            continue

        # === Load and combine available slide features ===
        for i, path in enumerate(available_features):
            slide_features = torch.load(path, map_location='cpu')
            animal_feature = slide_features if i == 0 else torch.cat((animal_feature, slide_features), dim=0)

        # Save animal-level features
        torch.save(animal_feature, animal_feature_path)

        # === Progress logging ===
        counter += 1
        percent_completed = (counter / len(animal_slide_dict)) * 100
        print(f"Processed {counter}/{len(animal_slide_dict)} animals ({percent_completed:.2f}%)")

    # === Summary ===
    print("\n=== Aggregation Summary ===")
    print(f"Animals processed: {counter}")
    print(f"Animals skipped (no features): {skipped_animals}")
    print(f"Animal features saved in: {animal_feature_dir}")


if __name__ == "__main__":
    # Load config
    config_path = "configs/configs.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)


    # Run the aggregation
    combine_slide_features_by_animal_organ_UID(config)

