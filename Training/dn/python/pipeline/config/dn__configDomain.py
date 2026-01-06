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

ext  = "Training/dataObj/original?_1/DValue_0.024720000000000002/rValue_1.536/gamma_0.2/K_1700.0/xNum_38/tNum_5/originalDataObj.npy"
Notebooks_path = findUpwardPath('July17')  # find data path
data_orig_obj_FKPP = np.load(os.path.join(Notebooks_path,ext),allow_pickle=True).item(0)
x = data_orig_obj_FKPP.x
t = data_orig_obj_FKPP.t
K = 1#data_orig_obj_FKPP.K

# ==========================================================================================
xNum =38
tNum = 5
x = np.linspace(np.min(x),np.max(x),xNum)
t = np.linspace(np.min(t),np.max(t),tNum)
# ==========================================================================================