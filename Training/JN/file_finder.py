import os
import pandas as pd
def find_data_obj_files(start_dir, target_filename="data_obj.npy"):
    """
    Recursively finds all files named `data_obj.npy` in directories with structure X_Y.

    Parameters:
        start_dir (str): Root directory to start the search.
        target_filename (str): The name of the file to search for.

    Returns:
        List[str]: List of full paths to matching files.
    """
    matches = []
    for root, dirs, files in os.walk(start_dir):
        if target_filename in files:
            matches.append(os.path.join(root, target_filename))
    return matches


def paths_to_df(paths):
    """
    Converts a list of paths into a pandas DataFrame based on X_Y directory names.

    Parameters:
        paths (List[str]): List of paths to data_obj.npy files.

    Returns:
        pd.DataFrame: DataFrame where each row represents a file and columns are variables X.
    """
    records = []
    for path in paths:
        row = {'full_path': path}
        parts = os.path.normpath(path).split(os.sep)
        for part in parts:
            if '_' in part:
                try:
                    x, y = part.split('_', 1)
                    row[x] = y
                except ValueError:
                    continue  # Skip if the format doesn't match
        records.append(row)
    return pd.DataFrame(records)



def condense_df(df, filters):
    """
    Filters the DataFrame by matching specified values for variables.

    Parameters:
        df (pd.DataFrame): Original DataFrame.
        filters (dict): Dictionary with variable names as keys and desired values as values.

    Returns:
        pd.DataFrame: Filtered/condensed DataFrame.
    """
    filtered_df = df.copy()
    for key, value in filters.items():
        filtered_df = filtered_df[filtered_df[key] == str(value)]
    return filtered_df.reset_index(drop=True)

def print_path_components(path):
    components = os.path.normpath(path).split(os.sep)
    for part in components:
        print(part)
