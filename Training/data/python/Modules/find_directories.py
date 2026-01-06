
import os

def findUpwardPath(target_dirname, start_path=None):
    """
    Recursively search upward to find the first directory named `target_dirname`.

    Parameters:
        target_dirname (str): Name of the directory to search for.
        start_path (str): Directory to start from. Defaults to current working dir.

    Returns:
        str: Absolute path to the found directory.

    Raises:
        FileNotFoundError: If the directory is not found in any parent path.
    """
    current_path = os.path.abspath(start_path or os.getcwd())

    while True:
        candidate = os.path.join(current_path, target_dirname)
        if os.path.isdir(candidate):
            return candidate

        parent = os.path.dirname(current_path)
        if parent == current_path:
            raise FileNotFoundError(
                f"Could not find directory '{target_dirname}' upward from '{os.path.abspath(start_path or os.getcwd())}'"
            )

        current_path = parent


def findAllPaths(filename, start_path=None):
    """
    Recursively walks upward from start_path and finds all matching file paths
    (by name) that exist below each ancestor directory.

    Parameters:
        filename (str): The file name to search for (e.g., 'config.yaml').
        start_path (str, optional): Directory to start from. Defaults to cwd.

    Returns:
        list of str: List of absolute file paths matching the filename.
    """
    matches = []
    current_path = os.path.abspath(start_path or os.getcwd())

    visited = set()  # avoid redundant re-scans

    while True:
        if current_path in visited:
            break
        visited.add(current_path)

        # Search downward from this level
        for root, _, files in os.walk(current_path):
            if filename in files:
                matches.append(os.path.join(root, filename))

        # Move up one directory
        parent = os.path.dirname(current_path)
        if parent == current_path:
            break  # reached filesystem root
        current_path = parent

    return matches
