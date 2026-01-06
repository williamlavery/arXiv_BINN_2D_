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

Modules_path = findUpwardPath('Modules')        # find Modules path
components_path = findUpwardPath('components')  # find components path
config_path = findUpwardPath('config')          # find config path

ExecAllWithin(Modules_path)
ExecAllWithin(components_path)
ExecAllWithin(config_path)
Notebooks_path = findUpwardPath('July17')  # find data path
#data_orig_path = findAllPaths("fisher_KPP_data.npy", start_path=Notebooks_path)
#ext  = "Training/dataObj/original?_1/DValue_0.024720000000000002/rValue_1.536/gamma_0.2/K_1700.0/xNum_38/tNum_5/originalDataObj.npy"

print(os.path.join(Notebooks_path,ext))

file_path = os.path.join(Notebooks_path, ext)

if os.path.isfile(file_path):
    print(f"✅ File found at: {file_path}")
else:
    print(f"❌ No file found at: {file_path}")


data_orig_obj_FKPP = np.load(os.path.join(Notebooks_path,ext),allow_pickle=True).item(0)

x = data_orig_obj_FKPP.x
t = data_orig_obj_FKPP.t
K = 1#data_orig_obj_FKPP.K
inputs = generate_inputs(x,t)
data_obj_path = findUpwardPath('dataObj') 
data_obj_path = findUpwardPath('dataObj') 

# Generate the clean solution 
data_obj_params = {}
data_obj_params["RDEq_params"] = {}
data_obj_params["add_noise_params"] = {}
data_obj_params["x"] = x
data_obj_params["t"] = t
data_obj_params["K"] = K

data_obj_params["RDEq_params"]["IC_label"] = IC_label
data_obj_params["add_noise_params"]["seed"] = seed
data_obj_params["add_noise_params"]["gamma"] = gamma


data_obj_params["inital_path"] = data_obj_path
data_obj_params["denoisePath"] = findUpwardPath('dn')
data_obj_params["RDEq_params"]["inputs"] = inputs
data_obj_params["plot_bool"] = plot_bool
data_obj_params["overwrite_bool"] = overwrite_bool

# ==========================================================================================
# ==========================================================================================
# ==========================================================================================
# ==========================================================================================

denoiseGenerate_indices_func = random_val_indices
denoiseGenerate_indices_func_info = random_val_indices_func_info
denoiseGenerate_indices_func_additional_args = [0]
denoiseGenerate_indices_func_label = "random"
denoiseTV_params = {}
denoiseTV_params["denoiseGenerate_indices_func"]              =   denoiseGenerate_indices_func
denoiseTV_params["denoiseGenerate_indices_func_info"]         =   denoiseGenerate_indices_func_info
denoiseTV_params["denoiseGenerate_indices_func_additional_args"]= denoiseGenerate_indices_func_additional_args

model_params = {}
model_params["denoiseDevice"]    = denoiseDevice
fit_params = {"denoiseTV_params": denoiseTV_params}
fit_params["denoiseLR"]                = denoiseLR
fit_params["denoiseBatchSize"]        = denoiseBatchSize
fit_params["denoiseRelUpdateThresh"] = denoiseRelUpdateThresh
fit_params["denoiseRelSaveThresh"]   = denoiseRelSaveThresh


# ==========================================================================================
# ==========================================================================================


total = len(denoiseVFs) * len(denoiseESs) * len(denoiseEpochs_lst) 
total *= len(diff_labels) * len(grow_labels) * len(noise_percents)
total *= len(denoiseUsizes) * len(denoiseModelLabels)
run_number = 0

# RUN
for denoiseVF in denoiseVFs:  # 1
    for denoiseES in denoiseESs: # 2 
        for diff_label in diff_labels: # 4
            for grow_label in grow_labels: # 5
                for noise_percent in noise_percents: # 6
                    for denoiseUsize in denoiseUsizes: # 7s
                        for denoiseModelLabel in denoiseModelLabels: # 8

                                    print("-" * 50)
                                    print(f"\n RUN {run_number}/{total} — Current settings:")
                                    print(f"  IC_label           = {IC_label}")
                                    print(f"  diff_label         = {diff_label}")
                                    print(f"  grow_label         = {grow_label}")
                                    print(f"  gamma              = {gamma}")
                                    print(f"  noise_percent      = {noise_percent}")
                                    print(f"  seed               = {seed}")
                                    print(f"  binnUsize          = {denoiseUsize}")
                                    print(f"  denoiseES          = {denoiseES}")
                                    print(f"  denoiseVF          = {denoiseVF}")
                                    print(f"   ModelLabel          = {denoiseModelLabel}")
                                    print("-" * 50)

                                    denoiseEpochs = denoiseEpochs_lst[denoiseModelLabel]

                                    model_params["denoiseModelLabel"]  = denoiseModelLabel #1
                                    model_params["denoiseUsize"]  = denoiseUsize #2
                                    data_obj_params["add_noise_params"]["noise_percent"] = noise_percent #3
                                    data_obj_params["RDEq_params"]["grow_label"]  = grow_label #4
                                    data_obj_params["RDEq_params"]["diff_label"]  = diff_label #5
                                    fit_params["denoiseEpochs"] = denoiseEpochs #6
                                    fit_params["denoiseES"]  = denoiseES #6
                                    fit_params ["denoiseVF"] = denoiseVF #8

                                    DN_sim(data_obj_params=data_obj_params, model_params=model_params, fit_params=fit_params)  
                                    run_number += 1
                                


