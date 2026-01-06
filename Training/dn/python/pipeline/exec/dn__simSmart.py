import sys, os, importlib.util, copy
from itertools import product
from math import prod

def print_run_summary(run_number, total, x, t, data_obj_params, TV_params, model_params, fit_params, omit_keys=None):
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
    print(f"  xNum{'':<21} = {len(x)}")
    print(f"  tNum{'':<21} = {len(t)}")

    # Grouped prints
    print("-" * 70)
    print("data_obj_params:")
    print_dict(data_obj_params)
    print("-" * 70)

    print("-" * 70)
    print("TV_params:")
    print_dict(TV_params)
    print("-" * 70)

    print("-" * 70)
    print("model_params:")
    print_dict(model_params)
    print("-" * 70)

    print("-" * 70)
    print("fit_params:")
    print_dict(fit_params)
    

    print("=" * 70)

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

Modules_path = findUpwardPath('Modules')        # find Modules path
components_path = findUpwardPath('components')  # find components path
config_path = findUpwardPath('config')          # find config path

ExecAllWithin(Modules_path)
ExecAllWithin(components_path)
ExecAllWithin(config_path)

inputs = generate_inputs(x,t)
data_obj_path = findUpwardPath('dataObj') 

# Generate the clean solution 
data_obj_params = {}
additional_params = {}
RDEq_params_store = {}
RDEq_params = {}
add_noise_params = {}


RDEq_params_store["x"] = x
RDEq_params_store["t"] = t
RDEq_params_store["inputs"] = inputs
RDEq_params_store["K"] = K

additional_params["inital_path"] = data_obj_path
additional_params["binn_path"] = findUpwardPath('binn')
additional_params["denoise_path"] = findUpwardPath('dn')
additional_params["plot_bool"]=plot_bool
additional_params["overwrite_bool"]=overwrite_bool

# ----------------------------------------------------------------------------------------------
data_obj_params["RDEq_params_store"] = RDEq_params_store
data_obj_params["RDEq_params"] = RDEq_params
data_obj_params["additional_params"] = additional_params
data_obj_params["add_noise_params"] = add_noise_params
# ----------------------------------------------------------------------------------------------


# ==========================================================================================
# ==========================================================================================
# DENOISE
# ==========================================================================================
# ==========================================================================================

# ==========================================================================================
# TV
# ==========================================================================================
TV_params = {}
denoiseTV_params = {}
#denoiseSeed = 0
#denoiseGenerateIndicesFuncLabel  = "random"
denoiseTV_params["denoiseVF"]              =  denoiseVFs[0] # initialize value for dictionary ordering
denoiseTV_params["denoiseGenerateIndicesLabel"]= denoiseGenerateIndicesFuncLabel 
denoiseTV_params["denoiseGenerateIndicesArgs"] = {"denoiseTVsplitSeed": denoiseTVsplitSeeds[0]}
TV_params["denoiseTV_params"] = denoiseTV_params
# Need to dynamically extend denoiseTV_params !!!
# ==========================================================================================
# MODEL params
# ==========================================================================================
model_params = {}
denoise_model_params = {}

# ==========================================================================================
# construction params
# ==========================================================================================
denoise_construction_params = {}
#denoise_construction_params["denoiseModelLabel"] = 0  # intialize for ordering
denoise_construction_params["denoiseUsize"] = 0
denoise_construction_params["denoiseDevice"]       =    denoiseDevice
denoise_model_params["denoise_construction_params"] = denoise_construction_params

# ==========================================================================================
# loss params
# ==========================================================================================
DNdata_loss_params = {}
DNdata_loss_params["DNdataLossFuncLabel"] = DNdataLossFuncLabel
denoise_model_params["DNdata_loss_params"] = DNdata_loss_params


# ----------------------------------------------------------------------------------------------
model_params["denoise_model_params"] = denoise_model_params
# ----------------------------------------------------------------------------------------------

# ==========================================================================================
# FIT params
# ==========================================================================================
fit_params = {}
denoise_fit_params_additionals = {}
denoise_fit_params = {}

denoise_fit_params["denoiseLR"]                = denoiseLR
denoise_fit_params["denoiseBatchSize"]        = denoiseBatchSize
denoise_fit_params["denoiseRelUpdateThresh"] = denoiseRelUpdateThresh
denoise_fit_params["denoiseRelSaveThresh"]   = denoiseRelSaveThresh
denoise_fit_params["denoiseES"]   = int(denoiseEpochs_lst[0])
denoise_fit_params["denoiseManualTermination"] = denoiseManualTermination
denoise_fit_params["denoiseModelLabel"] = 0  # intialize for ordering

denoise_fit_params_additionals["denoiseEpochs"]            =   0  # placeholder
denoise_fit_params_additionals["combineDenoise"] = combineDenoise

# ----------------------------------------------------------------------------------------------
fit_params["denoise_fit_params"] = denoise_fit_params
fit_params["denoise_fit_params_additionals"] = denoise_fit_params_additionals
# ----------------------------------------------------------------------------------------------


# ==========================================================================================
# ==========================================================================================


# Create parameter grid
param_lists =[
    denoiseVFs,         # 1
    denoiseESs,         # 2
    dataDiffLabels,        # 3
    dataGrowLabels,        # 4
    dataNoisePercents,     # 5
    denoiseUsizes,      # 6
    denoiseModelLabels, # 7
    denoiseTVsplitSeeds,   # 8
    dataGammas,
    dataNoiseSeeds
]

param_grid = product(*param_lists)

# Total number of runs
total = prod(len(lst) for lst in param_lists)

run_number = 0

for (
    denoiseVF,
    denoiseES,
    dataDiffLabel,
    dataGrowLabel,
    dataNoisePercent,
    denoiseUsize,
    denoiseModelLabel,
    denoiseTVsplitSeed,
    dataGamma,
    dataNoiseSeed
) in param_grid:


    # update dn Epochs
    # ===========================
    denoiseEpochs = denoiseEpochs_lst[denoiseModelLabel]
    fit_params["denoise_fit_params_additionals"]['denoiseEpochs'] = int(denoiseEpochs)
    # -----------------------------

    # update dn construction
    # ===========================
    model_params["denoise_model_params"]["denoise_construction_params"]["denoiseUsize"] = denoiseUsize 
    #model_params["denoise_model_params"] = denoise_model_params
    # -----------------------------

    # update data object params
    # ===========================
    data_obj_params["RDEq_params"]["dataICLabel"] = dataICLabel
    data_obj_params["RDEq_params"]["dataDiffLabel"] = dataDiffLabel
    data_obj_params["RDEq_params"]["dataGrowLabel"] = dataGrowLabel
    # -----------------------------

    add_noise_params =  data_obj_params["add_noise_params"]
    add_noise_params["dataNoisePercent"] = dataNoisePercent
    add_noise_params["dataGamma"] = dataGamma
    add_noise_params["dataNoiseSeed"] = dataNoiseSeed
    ##
    data_obj_params["add_noise_params"] = add_noise_params
    # -----------------------------

    # update binn ES
    # ===========================
    fit_params["denoise_fit_params"]['denoiseES'] = denoiseES
    # -----------------------------

    # update dn TV params
    # ===========================
    TV_params["denoiseTV_params"]['denoiseVF'] = denoiseVF
    TV_params["denoiseTV_params"]["denoiseGenerateIndicesArgs"] = {"denoiseTVsplitSeed": denoiseTVsplitSeed}
    ##
    # -----------------------------

    print_run_summary(
    run_number=run_number,
    total=total,
    x=x,
    t=t,
    data_obj_params=data_obj_params,
    TV_params=TV_params,
    model_params=model_params,
    fit_params=fit_params,
    omit_keys=["x", "t", "inputs"]  # skip these
    )

    DN_sim_smart(
    data_obj_params=data_obj_params,
    TV_params =TV_params, 
    model_params=model_params,
    fit_params=fit_params
    )

    run_number += 1

