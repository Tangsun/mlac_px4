#!/usr/bin/env python3

import pickle
import argparse
import os

def clean_pkl_file(source_path, destination_path, keys_to_keep):
    """
    Loads a dictionary from a source .pkl file, filters it to keep only specified keys,
    and saves the cleaned dictionary to a new .pkl file.

    Args:
        source_path (str): Path to the source .pkl file.
        destination_path (str): Path to save the cleaned .pkl file.
        keys_to_keep (list): A list of strings representing the keys to retain.
    """
    try:
        print(f"Attempting to load data from: {source_path}")
        with open(source_path, 'rb') as f_in:
            data = pickle.load(f_in)
        print(f"Successfully loaded data from: {source_path}")
    except FileNotFoundError:
        print(f"Error: Source file '{source_path}' not found.")
        return
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle data from '{source_path}'. "
              "The file might be corrupted or not a valid pickle file.")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading '{source_path}': {e}")
        return

    if not isinstance(data, dict):
        print(f"Error: Expected a dictionary in '{source_path}', but found type {type(data)}.")
        return

    print(f"Original dictionary contains {len(data)} keys. First few keys: {list(data.keys())[:5]}...")

    cleaned_data = {}
    kept_keys_count = 0
    missing_keys_list = []

    for key in keys_to_keep:
        if key in data:
            cleaned_data[key] = data[key]
            kept_keys_count += 1
            print(f"  Keeping key: '{key}'")
        else:
            missing_keys_list.append(key)
            print(f"  Warning: Key '{key}' not found in the source dictionary.")

    if kept_keys_count == 0 and missing_keys_list:
        print(f"Warning: None of the specified keys ({', '.join(keys_to_keep)}) were found. "
              "The output file will contain an empty dictionary.")
    elif missing_keys_list:
        print(f"Summary: Kept {kept_keys_count} key(s). "
              f"Specified keys not found: {', '.join(missing_keys_list)}")
    else:
        print(f"Successfully identified all {kept_keys_count} specified key(s) to keep.")

    # Ensure the destination directory exists
    try:
        dest_dir = os.path.dirname(destination_path)
        if dest_dir and not os.path.exists(dest_dir): # Check if dest_dir is not empty (e.g. for relative paths in current dir)
            os.makedirs(dest_dir)
            print(f"Created destination directory: {dest_dir}")
    except Exception as e:
        print(f"Error creating destination directory for '{destination_path}': {e}")
        return

    try:
        print(f"Attempting to save cleaned dictionary with {len(cleaned_data)} keys to: {destination_path}")
        with open(destination_path, 'wb') as f_out:
            pickle.dump(cleaned_data, f_out)
        print(f"Successfully saved cleaned data to: {destination_path}")
    except Exception as e:
        print(f"An error occurred while saving the cleaned data to '{destination_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cleans a .pkl file by extracting a subset of keys from its root dictionary.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "source_path",
        type=str,
        help="Path to the source .pkl file."
    )
    parser.add_argument(
        "destination_path",
        type=str,
        help="Path where the cleaned .pkl file will be saved."
    )
    # The keys to keep are fixed as per the request, but could be made an argument
    # parser.add_argument(
    #     "--keys",
    #     nargs='+',
    #     default=["pnorm", "model", "controller"],
    #     help="List of keys to keep in the dictionary. Default: pnorm model controller"
    # )

    args = parser.parse_args()

    # Fixed keys as per your request
    keys_to_keep = ["pnorm", "model", "controller"]

    clean_pkl_file(args.source_path, args.destination_path, keys_to_keep)