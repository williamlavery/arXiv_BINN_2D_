# ================ Training and validation data splitting ================ 



def DN_random_val_indices(inputs, VF, denoiseGenerateIndicesArgs):

    seed=denoiseGenerateIndicesArgs["denoiseTVsplitSeed"]
    N = len(inputs)
    num_val = int(VF * N)
    # Generate random permutation of indices
    np.random.seed(seed)
    p = np.random.permutation(N)
    # Take the last portion as validation indices
    val_idx = p[-num_val:]
                
    return val_idx

def DN_TVsplit_func_info(TV_params):
    denoiseTV_params = TV_params["denoiseTV_params"]
    denoiseVF = denoiseTV_params["denoiseVF"]
    if denoiseVF:
        return unravel_one_level(denoiseTV_params)
    return {"denoiseVF":denoiseVF}



def DN_TVsplit(data_obj, TV_params, model_params):
    inputs = data_obj.inputs
    u = data_obj.u.copy()   # nosiy density
    u_clean_np = data_obj.u_clean.copy() 
    outputs = u.reshape(-1)[:, None] 

    denoiseTV_params = TV_params["denoiseTV_params"]
    denoiseVF = denoiseTV_params["denoiseVF"]
    denoiseGenerateIndicesLabel = denoiseTV_params["denoiseGenerateIndicesLabel"]
    denoiseGenerateIndicesArgs = denoiseTV_params["denoiseGenerateIndicesArgs"]

    denoise_model_params = model_params["denoise_model_params"]
    denoise_construction_params = denoise_model_params["denoise_construction_params"]
    device =  denoise_construction_params["denoiseDevice"]

    if denoiseVF:
        if denoiseGenerateIndicesLabel== "random":
            val_indices = DN_random_val_indices(inputs, denoiseVF, denoiseGenerateIndicesArgs)

        # Remaining for training
        N = len(inputs)
        all_indices = np.arange(N)

        train_indices = np.setdiff1d(all_indices, val_indices)


        x_train_np = inputs[train_indices]
        y_train_np = outputs[train_indices]
        x_val_np = inputs[val_indices]
        y_val_np = outputs[val_indices]

        val_dic = {"x_val_np": x_val_np,
                    "y_val_np":y_val_np,
                    "validation_data":[x_val_np, y_val_np]}

    else:
    
        x_train_np = inputs
        y_train_np = outputs

        x_val_np = []
        y_val_np = []

        val_dic = {
            "x_val_np": x_val_np,
            "y_val_np":y_val_np,
            "validation_data":None}


    train_dic = {"x_train_np": x_train_np,
                 "y_train_np":y_train_np,
                 "u_clean_np":u_clean_np}


    TV_dic = {"train_dic":train_dic, "val_dic": val_dic}
    func_info = DN_TVsplit_func_info(TV_params=TV_params)
    return func_info, TV_dic