import sys, os, importlib.util


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

def ExecAllWithin(base_dir):
    """
    Recursively executes all Python files in a directory.
    All top-level definitions (functions, imports, etc.)
    will be available in the global namespace.

    Parameters:
        base_dir (str): Root directory to walk.
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        raise NotADirectoryError(f"'{base_dir}' is not a valid directory.")

    print(f"\n⚡ Executing all Python files in: {base_dir}\n")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                file_path = os.path.join(root, file)
                print(f"  ▶ Running: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    exec(code, globals())  # Execute directly in global scope

    print("\n✅ All files executed and symbols injected.\n")




# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
file_name = "originalDataObj.npy"

Modules_path = findUpwardPath('Modules')  # find data path
ExecAllWithin(Modules_path)

# file path intro
Data_path = findUpwardPath('DataStore')  # find data path
path_to_paper_data = os.path.join(Data_path,"paper/fisher_KPP_data.npy")  


data = np.load(path_to_paper_data, allow_pickle=True).item(0)
data_orig = OriginalData(data=data, plot=False)  # generate structure 



data_obj_path = findUpwardPath('dataObj') 
class_info = data_orig.class_info

save_path = os.path.join(data_obj_path, dictToPath(class_info))
os.makedirs(save_path, exist_ok=True)

print("saved file at:", os.path.join(save_path,file_name))
np.save(os.path.join(save_path,file_name), data_orig,allow_pickle=True)  # save structure
