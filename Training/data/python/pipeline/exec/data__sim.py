import sys, os, importlib.util, copy
from itertools import product
from math import prod

def print_run_summary(run_number, total, x1,x2, t, data_obj_params, omit_keys=None):
    """
    Pretty-print parameters grouped by their dictionary source, with option to omit specific keys.
    
    Args:
        omit_keys (list): List of keys (strings) to omit, at any depth.
    """
    omit_keys = set(omit_keys or [])

    def print_dict(d, indent=2):
        """Recursively print dictionary contents with indentation, skipping omitted keys."""
        for key, value in d.items():
            if key in omit_keys:
                continue  # Skip this key entirely

            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                print_dict(value, indent + 4)
            else:
                print(" " * indent + f"{key:<27} = {value}")

    print("=" * 70)
    print(f" RUN {run_number}/{total} — Current settings")
    print("=" * 70)

    # Quick base info
    print("Base Info:")
    print(f"  x1Num{'':<21} = {len(x1)}")
    print(f"  x2Num{'':<21} = {len(x2)}")
    print(f"  tNum{'':<21} = {len(t)}")

    # Grouped prints
    print("-" * 70)
    print("data_obj_params:")
    print_dict(data_obj_params)
    print("-" * 70)



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

def findDownwardPath(target_dirname, start_path=None):
    """
    Recursively search downward to find the first directory named `target_dirname`.

    Parameters:
        target_dirname (str): Name of the directory to search for.
        start_path (str): Directory to start from. Defaults to current working dir.

    Returns:
        str: Absolute path to the found directory.

    Raises:
        FileNotFoundError: If the directory is not found.
    """
    root_path = os.path.abspath(start_path or os.getcwd())

    for dirpath, dirnames, _ in os.walk(root_path):
        if target_dirname in dirnames:
            return os.path.join(dirpath, target_dirname)

    raise FileNotFoundError(
        f"Could not find directory '{target_dirname}' downward from '{root_path}'"
    )
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

    #print(f"\n⚡ Executing all Python files in: {base_dir}\n")

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                file_path = os.path.join(root, file)
                #print(f"  ▶ Running: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                    exec(code, globals())  # Execute directly in global scope

    #print("\n✅ All files executed and symbols injected.\n")


# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================

Modules_path = findUpwardPath('Modules')  # find data path
config_path = findUpwardPath('config')  # find data path
components_path = findUpwardPath('components')  # find data path


ExecAllWithin(Modules_path)
ExecAllWithin(components_path)
ExecAllWithin(config_path)

#data_orig_path = findAllPaths("fisher_KPP_data.npy", start_path=Notebooks_path)
data_obj_path = findUpwardPath('dataObj') 

# Generate the clean solution 
data_obj_params = {}
additional_params = {}
RDEq_params_store = {}
RDEq_params = {}
add_noise_params = {}

RDEq_params_store["x1"] = x1
RDEq_params_store["x2"] = x2
RDEq_params_store["t"] = t
RDEq_params_store["inputs"] = inputs
RDEq_params_store["K"] = K

additional_params["inital_path"] = data_obj_path
additional_params["binn_path"] = findUpwardPath('binn')
additional_params["plot_bool"]=plot_bool
additional_params["overwrite_bool"]=overwrite_bool


RDEq_params["dataX1num"] = dataX1num
RDEq_params["dataX2num"] = dataX2num
RDEq_params["dataTnum"] =  dataTnum
RDEq_params["dataK"] =  K


# ----------------------------------------------------------------------------------------------
#data_obj_params["fundamental_params"] = fundamental_params
data_obj_params["RDEq_params_store"] = RDEq_params_store
data_obj_params["RDEq_params"] = RDEq_params
data_obj_params["additional_params"] = additional_params
data_obj_params["add_noise_params"] = add_noise_params


# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================

param_lists = [
dataGammas,
dataNoisePercents,
dataNoiseSeeds,
dataICLabels,
dataDiffLabels,
dataGrowLabels,
]


param_grid = product(*param_lists)

# Total number of runs
total = prod(len(lst) for lst in param_lists)

run_number = 1  # initialize run counter
for (dataGamma,
     dataNoisePercent,
     dataNoiseSeed,
     dataICLabel,
     dataDiffLabel,
     dataGrowLabel
) in param_grid:
    
    data_obj_params["add_noise_params"]["dataGamma"]     = dataGamma
    data_obj_params["add_noise_params"]["dataNoisePercent"] = dataNoisePercent
    data_obj_params["add_noise_params"]["dataNoiseSeed"]      = dataNoiseSeed  # Note: original mapped "u_clean" to "dataNoiseSeed"     
    data_obj_params["RDEq_params"]["dataICLabel"]    = dataICLabel
    data_obj_params["RDEq_params"]["dataDiffLabel"]  = dataDiffLabel
    data_obj_params["RDEq_params"]["dataGrowLabel"]  = dataGrowLabel

    print_run_summary(run_number, total, x1, x2, t, data_obj_params, omit_keys=None)
    
    # RUN ALL PIPELINE COMPONENTS
    DATA_sim(data_obj_params)

    run_number += 1
                        
       

# TRUE
# dataDiffLabel_exp/dataGrowLabel_zero/datadataGamma_1/dataNoisePercent_0/dataNoisedataNoiseSeed_0/data_obj.npy

# MINE
# dataDiffLabel_const/dataGrowLabel_zero/datadataGamma_1/dataNoisePercent_0/dataNoisedataNoiseSeed_0/data_obj.npy
