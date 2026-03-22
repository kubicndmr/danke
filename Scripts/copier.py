import os
import shutil

from pathlib import Path


def get_indices(target_dir, prefix):
    indices = []

    # Iterate through all CSV files
    for item in target_dir.glob("*.csv"):
        
        # Check if txt counterpart exists
        assert os.path.isfile(item.with_suffix(".txt"))
        
        # Extract index from filename
        try:
            item = os.path.basename(item)
            assert item.endswith(".csv")
            index = int(item[len(prefix):-4])
            indices.append(index)
        except ValueError:
            continue
    
    indices = sorted(indices)

    # Check if indices are countiniously increasing
    for i in range(len(indices) - 1):
        assert indices[i] + 1 == indices[i + 1], f"Indices are not continuous at {indices[i]} and {indices[i+1]}"
    
    return indices


if __name__ == '__main__':
    source = Path("SynPoCaP-17")
    destination = Path("TextualSPR-Dataset/SynPoCaP")
    prefix = "SynOP_"

    if not os.path.exists(source):
        raise FileNotFoundError(f"Source directory '{source}' does not exist.")

    # Check if all csv files has corrsponding txt files in source directory
    source_indices = get_indices(source, prefix)
    destination_indices = get_indices(destination, prefix)

    # Find the last index in the source directory
    last_index = max(destination_indices) + 1

    # Copy the source directory to the destination
    for i, item in enumerate(source.glob("*.csv")):
        item_name = f"{prefix}{last_index + i:05d}"
        
        destination_item_csv = destination / (item_name + ".csv")
        shutil.copy(item, destination_item_csv)
        print(f"Moved {item} to {destination_item_csv}")

        destination_item_txt = destination / (item_name + ".txt")
        shutil.copy(item.with_suffix('.txt'), destination_item_txt)
        print(f"Moved {item.with_suffix('.txt')} to {destination_item_txt}\n")