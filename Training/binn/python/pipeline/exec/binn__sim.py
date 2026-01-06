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
dn_path = findUpwardPath('dn') 
components_dn_path = os.path.join(dn_path, "python/pipeline/components") 


ExecAllWithin(Modules_path)
ExecAllWithin(components_path)
ExecAllWithin(components_dn_path)
ExecAllWithin(config_path)


Notebooks_path = findUpwardPath('July17')  # find data path
#data_orig_path = findAllPaths("fisher_KPP_data.npy", start_path=Notebooks_path)
ext  = "Training/dataObj/original?_1/DValue_0.024720000000000002/rValue_1.536/gamma_0.2/K_1700.0/xNum_38/tNum_5/originalDataObj.npy"

#print(os.path.join(Notebooks_path,ext))

file_path = os.path.join(Notebooks_path, ext)

#if os.path.isfile(file_path):
   # print(f"✅ File found at: {file_path}")
#else:
   # print(f"❌ No file found at: {file_path}")


data_orig_obj_FKPP = np.load(os.path.join(Notebooks_path,ext),allow_pickle=True).item(0)

x = data_orig_obj_FKPP.x
t = data_orig_obj_FKPP.t
K = 1#data_orig_obj_FKPP.K#1
#u = data_orig_obj_FKPP.u
inputs = generate_inputs(x,t)
data_obj_path = findUpwardPath('dataObj') 

# Generate the clean solution 
data_obj_params = {}
data_obj_params["RDEq_params"] = {}
data_obj_params["add_noise_params"] = {}
data_obj_params["x"] = x
data_obj_params["t"] = t
data_obj_params["K"] = K
#data_obj_params["u"] = u
data_obj_params["inital_path"] = data_obj_path
data_obj_params["binnPath"] = findUpwardPath('binn')
data_obj_params["RDEq_params"]["inputs"] = inputs
data_obj_params["denoisePath"] = findUpwardPath('dn')

data_obj_params["plot_bool"]=plot_bool
data_obj_params["overwrite_bool"]=overwrite_bool

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================

denoiseGenerate_indices_func = DN_random_val_indices
denoiseGenerate_indices_func_info = DN_random_val_indices_func_info
#denoiseGenerate_indices_func_additional_args_lst = denoiseModelLabels
denoiseGenerate_indices_func_label = "random"
denoiseTV_params = {}
denoiseTV_params["denoiseGenerate_indices_func"]              =   denoiseGenerate_indices_func
denoiseTV_params["denoiseGenerate_indices_func_info"]         =   denoiseGenerate_indices_func_info
#denoiseTV_params["denoiseGenerate_indices_func_additional_args"]= denoiseGenerate_indices_func_additional_args_lst[0]
denoiseTV_params["denoiseGenerate_indices_func_label"]= denoiseGenerate_indices_func_label 

model_params = {}
model_params["denoiseDevice"]    = denoiseDevice
fit_params = {"denoiseTV_params": denoiseTV_params}
fit_params["denoiseLR"]                = denoiseLR
fit_params["denoiseBatchSize"]        = denoiseBatchSize
fit_params["denoiseRelUpdateThresh"] = denoiseRelUpdateThresh
fit_params["denoiseRelSaveThresh"]   = denoiseRelSaveThresh
fit_params["denoiseVF"]   = denoiseVF
fit_params["denoiseES"]   = denoiseES
# ==========================================================================================
# ==========================================================================================

binnGenerate_indices_func = BN_random_val_indices
binnGenerate_indices_func_info = BN_random_val_indices_func_info
binnGenerate_indices_func_additional_args_lst = binnSplitSeeds
binnGenerate_indices_func_label = "random"
binnTV_params = {}
binnTV_params["binnGenerate_indices_func"]              =  binnGenerate_indices_func
binnTV_params["binnGenerate_indices_func_info"]         =   binnGenerate_indices_func_info
#binnTV_params["binnGenerate_indices_func_additional_args"]= binnGenerate_indices_func_additional_args
binnTV_params["binnGenerate_indices_func_label"]          = binnGenerate_indices_func_label

model_params["D_one_param_bool"] = D_one_param_bool
model_params["binnDevice"]       =     binnDevice
fit_params["binnTV_params"] = binnTV_params
fit_params["two_step_bool"]        = two_step_bool 
fit_params["binnLR"]                =    binnLR
fit_params["binnBatchSize"]        =    binnBatchSize
fit_params["binnRelUpdateThresh"] =    binnRelUpdateThresh
fit_params["binnRelSaveThresh"]   =    binnRelSaveThresh
fit_params["binnEpochs"]            =    binnEpochs
# ==========================================================================================
# ==========================================================================================


# RUN
       
total = len(gammas) * len(noise_percents) * len(diffLables) * len(growLabels) * len(binnUsizes) * len(binnDsizes) * len(binnGsizes) * len(binnESs) * len(binnVFs)
run_number = 0 # initialize run counter

from itertools import product


# Define the parameter grid
param_grid = product(
    gammas,
    noise_percents,
    noise_seeds,
    ICLabels,
    diffLables,
    growLabels,
    binnUsizes,
    binnDsizes,
    binnGsizes,
    binnESs,
    binnSplitSeeds,
    binnVFs,
    binnModelLabels
)

# Total number of runs
total = len(gammas) * len(noise_percents) * len(noise_seeds) * len(ICLabels) \
      * len(diffLables) * len(growLabels) * len(binnUsizes) * len(binnDsizes) \
      * len(binnGsizes) * len(binnESs) * len(binnESs) * len(binnSplitSeeds) * len(binnModelLabels)

run_number = 0

# Loop over all combinations
for (
    gamma, noise_percent, noise_seed, IC_label,
    diff_label, grow_label, binnUsize,
    binnDsize, binnGsize, binnES, binnSplitSeed, binnVF,  
    binnModelLabel
) in param_grid:
    
    model_params["binnModelLabel"] = binnModelLabel
    # Update model and fit params
    model_params["binnUsize"] = binnUsize
    model_params["binnDsize"] = binnDsize
    model_params["binnGsize"] = binnGsize

    # use same denoise and model seeds
    model_params["denoiseModelLabel"] = binnModelLabel
    model_params["denoiseUsize"] = binnUsize

    fit_params['denoiseEpochs'] = denoiseEpochs_lst[model_params["denoiseModelLabel"]]
    fit_params["binnES"] = binnES
    fit_params["binnVF"] = binnVF

    # Print run summary
    print("-" * 50)
    print(f"\n RUN {run_number}/{total} — Current settings:")
    print(f"  IC_label           = {IC_label}")
    print(f"  diff_label         = {diff_label}")
    print(f"  grow_label         = {grow_label}")
    print(f"  gamma              = {gamma}")
    print(f"  noise_percent      = {noise_percent}")
    print(f"  noise_seed         = {noise_seed}")
    print(f"  binnUsize          = {binnUsize}")
    print(f"  binnDsize          = {binnDsize}")
    print(f"  binnGsize          = {binnGsize}")
    print(f"  binnES             = {binnES}")
    print(f"  binnSplitSeed      = {binnSplitSeed}")
    print(f"  binnVF             = {binnVF}")
    print(f"  binnLR             = {fit_params['binnLR']}")
    print(f"  binnBatchSize      = {fit_params['binnBatchSize']}")
    print(f"  binnRelSaveThresh  = {fit_params['binnRelSaveThresh']}")
    print(f"  binnRelUpdateThresh= {fit_params['binnRelUpdateThresh']}")
    print(f"  binnEpochs         = {fit_params['binnEpochs']}")
    print(f"  D_one_param_bool   = {D_one_param_bool}")
    print(f"  device             = {model_params['binnDevice']}")
    print(f"  binnModelLabel        = {model_params['binnModelLabel']}")
    print("-" * 50)

    # Update data object params
    data_obj_params["RDEq_params"]["IC_label"] = IC_label
    data_obj_params["RDEq_params"]["diff_label"] = diff_label
    data_obj_params["RDEq_params"]["grow_label"] = grow_label

    data_obj_params["add_noise_params"]["gamma"] = gamma
    data_obj_params["add_noise_params"]["noise_percent"] = noise_percent
    data_obj_params["add_noise_params"]["seed"] = noise_seed

    fit_params["binnTV_params"]["binnGenerate_indices_func_additional_args"]= [binnSplitSeed]


    # Run
    BN_sim(
        data_obj_params=data_obj_params,
        model_params=model_params,
        fit_params=fit_params
    )

    run_number += 1


