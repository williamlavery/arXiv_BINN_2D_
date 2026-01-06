import sys, os, importlib.util, copy
from itertools import product
from math import prod

def print_run_summary(run_number, total, x1, x2, t, data_obj_params, TV_params, model_params, fit_params, omit_keys=None):
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

Modules_path = findUpwardPath('Modules')        # find Modules path
components_path = findUpwardPath('components')  # find components path
config_path = findUpwardPath('config')          # find config path
denoise_path = findUpwardPath('dn') 
components_denoise_path = os.path.join(denoise_path, "python/pipeline/components") 
Modules_denoise_path = os.path.join(denoise_path, "python/Modules/") 


ExecAllWithin(Modules_denoise_path)
ExecAllWithin(Modules_path)
ExecAllWithin(components_path)
ExecAllWithin(config_path)


#data_orig_path = findAllPaths("fisher_KPP_data.npy", start_path=Notebooks_path)
#u = data_orig_obj_FKPP.u

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
RDEq_params_store["inputs"] =generate_inputs_2d(x1,x2, t)
RDEq_params_store["K"] = K



additional_params["inital_path"] = data_obj_path
additional_params["binn_path"] = findUpwardPath('binn')
additional_params["denoise_path"] = findUpwardPath('dn')
additional_params["plot_bool"]=plot_bool
additional_params["overwrite_bool"]=overwrite_bool

RDEq_params["dataX1num"] = len(x1)
RDEq_params["dataX2num"] = len(x2)
RDEq_params["dataTnum"] =  len(t)
RDEq_params["dataK"] =  K

# ----------------------------------------------------------------------------------------------
#data_obj_params["fundamental_params"] = fundamental_params
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
denoiseTV_params["denoiseVF"]              =  denoiseVF # initialize value for dictionary ordering
denoiseTV_params["denoiseGenerateIndicesLabel"]= denoiseGenerateIndicesFuncLabel 
denoiseTV_params["denoiseGenerateIndicesArgs"] = {"denoiseTVsplitSeed": denoiseTVsplitSeeds[0]} # intialize
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
denoise_fit_params["denoiseES"]   = denoiseES
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
# Model
# ==========================================================================================
# ==========================================================================================

# ==========================================================================================
# TV
# ==========================================================================================
binnTV_params = {}
binnTV_params["binnVF"]              =  0.2 # initialize value for dictionary ordering
binnTV_params["binnGenerateIndicesLabel"]         =   binnGenerateIndicesLabel
binnGenerateIndicesArgs= {"binnTVsplitSeed": 0} # initialize value for dictionary ordering
binnTV_params["binnGenerateIndicesArgs"] =  binnGenerateIndicesArgs
TV_params["binnTV_params"] = binnTV_params


# ==========================================================================================
# MODEL params
# ==========================================================================================

# ==========================================================================================
# construction params
# ==========================================================================================
binn_model_params = {}
binn_construction_params = {}
binn_construction_params["binnUsize"] = 0
binn_construction_params["binnDsize"] = 0
binn_construction_params["binnGsize"] = 0 

binn_construction_params["DoneParamBool"] = DoneParamBool
binn_construction_params["binnDevice"]       =     binnDevice
binn_construction_params["binnInitializeDenoiseBool"] = binnInitializeDenoiseBool
binn_construction_params["allConstraints"] = allConstraints[0]  
binn_model_params["binn_construction_params"] = binn_construction_params
# ==========================================================================================
# loss params
# ==========================================================================================
BNdata_loss_params = {}
BNdata_loss_params["BNdataLossFuncLabel"] = BNdataLossFuncLabels[0]
binn_model_params["BNdata_loss_params"] = BNdata_loss_params

pde_loss_params = {}
pde_loss_params["BCbool"] = BCbool
pde_loss_params["numPDEsamples"]  = numPDEsamples
binn_model_params["pde_loss_params"] = pde_loss_params

# ----------------------------------------------------------------------------------------------
model_params["binn_model_params"] = binn_model_params
# ----------------------------------------------------------------------------------------------




# ==========================================================================================
# fit params
# ==========================================================================================
binn_fit_params = {}
binn_fit_params_additionals = {}

binn_fit_params["twoStepBool"]        = twoStepBools[0]
binn_fit_params["binnLR"]                =    binnLR
binn_fit_params["binnBatchSize"]        =   len(x1)*len(x2)
binn_fit_params["binnRelUpdateThresh"] =    binnRelUpdateThresh
binn_fit_params["binnRelSaveThresh"]   =    binnRelSaveThresh
binn_fit_params["binnES"] = binnESs[0] # order initializing
binn_fit_params["combineDenoise"] = combineDenoise
binn_fit_params["binnModelLabel"] = 0
binn_fit_params_additionals["binnEpochs"]            =    binnEpochs
binn_fit_params_additionals["binnES_check"]            =   binnES_check
binn_fit_params_additionals["printFreq"] = printFreq

# ----------------------------------------------------------------------------------------------
fit_params["binn_fit_params"] = binn_fit_params
fit_params["binn_fit_params_additionals"] = binn_fit_params_additionals
# ----------------------------------------------------------------------------------------------

# ==========================================================================================
# ==========================================================================================


# RUN
prev_binnTVsplitSeed = 0
idx_binnTVsplitSeed = 0  # start at -1 so first increment gives 0

# Define the parameter grid
param_lists = [
    gammas,
    noisePercents,
    noiseSeeds,
    ICLabels,
    diffLabels,
    growLabels,
    binnUsizes,
    binnDsizes,
    binnGsizes,
    binnESs,
    binnTVsplitSeeds,
    binnVFs,
    binnModelLabels,
    numPDEsamples,
    BNdataLossFuncLabels,
    allConstraints,
    ts,
    twoStepBools
]


param_grid = product(*param_lists)

# Total number of runs
total = prod(len(lst) for lst in param_lists)
print("Total runs to execute:", total)

run_number = 1

# Loop over all combinations
for (
    dataGamma, 
    dataNoisePercent, 
    dataNoiseSeed, 
    dataICLabel,
    dataDiffLabel,
    dataGrowLabel,
    binnUsize,
    binnDsize, 
    binnGsize, 
    binnES, 
    binnTVsplitSeed, 
    binnVF,  
    binnModelLabel,
    numPDEsample,
    BNdataLossFuncLabel,
    allConstraint,
    t,
    twoStepBool
) in param_grid:

    RDEq_params_store["t"] = t
    RDEq_params["dataTnum"] = len(t)
    

    # update binn construction
    # ===========================
    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]
    pde_loss_params = binn_model_params["pde_loss_params"] 

    model_params["binn_model_params"]["binn_construction_params"]["binnUsize"] = binnUsize
    model_params["binn_model_params"]["binn_construction_params"]["binnDsize"] = binnDsize
    model_params["binn_model_params"]["binn_construction_params"]["binnGsize"] = binnGsize
    model_params["binn_model_params"]["pde_loss_params"]["numPDEsamples"]  = numPDEsample
    model_params["binn_model_params"]["BNdata_loss_params"]["BNdataLossFuncLabel"] = BNdataLossFuncLabel
    model_params["binn_model_params"]["binn_construction_params"]["allConstraints"] = allConstraint
    # -----------------------------


    # update binn ES
    # ===========================
    fit_params["binn_fit_params"]["binnES"] = binnES
    fit_params["binn_fit_params"]["binnModelLabel"] = binnModelLabel
    # -----------------------------

    # update data object params
    # ===========================
    data_obj_params["RDEq_params"]["dataICLabel"] = dataICLabel
    data_obj_params["RDEq_params"]["dataDiffLabel"] = dataDiffLabel
    data_obj_params["RDEq_params"]["dataGrowLabel"] = dataGrowLabel
    # -----------------------------
    data_obj_params["add_noise_params"]["dataGamma"] = dataGamma
    data_obj_params["add_noise_params"]["dataNoisePercent"] = dataNoisePercent
    data_obj_params["add_noise_params"]["dataNoiseSeed"] = dataNoiseSeed
    # -----------------------------

    # update binn TV params
    # ===========================
    TV_params["binnTV_params"]["binnVF"] = binnVF
    TV_params["binnTV_params"]["binnGenerateIndicesArgs"] = {"binnTVsplitSeed": binnTVsplitSeed}
    # -----------------------------
    
    fit_params["binn_fit_params"]["twoStepBool"] = twoStepBool



    if fit_params["binn_fit_params"]["twoStepBool"] or binn_construction_params["binnInitializeDenoiseBool"]:

        model_params["binn_model_params"]["binn_construction_params"]["binnInitializeDenoiseBool"]  = 1
        # update dn construction
        # ===========================
        # use same denoise and model seeds
        model_params["denoise_model_params"]["denoise_construction_params"]["denoiseUsize"] = binnUsize
        # -----------------------------

        # update dn Epochs
        # ===========================
        fit_params["denoise_fit_params"]["denoiseModelLabel"] = binnModelLabel
        fit_params["denoise_fit_params_additionals"]['denoiseEpochs'] = denoiseEpochs_lst[binnModelLabel]
        # -----------------------------
    
        if binnTVsplitSeed != prev_binnTVsplitSeed:
            idx_binnTVsplitSeed += 1
            idx_binnTVsplitSeed = idx_binnTVsplitSeed%len(binnTVsplitSeeds)
            prev_binnTVsplitSeed = binnTVsplitSeed

        # update dn TV params
        # ===========================
        denoiseTVsplitSeed = denoiseTVsplitSeeds[idx_binnTVsplitSeed]
        TV_params["denoiseTV_params"]["denoiseGenerateIndicesArgs"] = {"denoiseTVsplitSeed": denoiseTVsplitSeed}
        # -----------------------------
    #else:
     #   model_params["binn_model_params"]["binn_construction_params"]["binnInitializeDenoiseBool"] = 0


    print_run_summary(
    run_number=run_number,
    total=total,
    x1=x1,
    x2=x2,
    t=t,
    data_obj_params=data_obj_params,
    TV_params=TV_params,
    model_params=model_params,
    fit_params=fit_params,
    omit_keys=["x", "t", "inputs", "u_clean"]  # skip these
    )


    # Run
    BN_sim_smart(
    data_obj_params=data_obj_params,
    TV_params=TV_params,
    model_params=model_params,
    fit_params=fit_params
    )

    run_number += 1


