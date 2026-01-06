# ============================ Load the data in to processs  ============================

def BN_load_func_info(data_obj_params, TV_params, model_params, fit_params):
    binn_fit_params = fit_params["binn_fit_params"]
    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]

    if binn_fit_params["twoStepBool"] or binn_construction_params["binnInitializeDenoiseBool"]:
        func_info = BN_load_dn_func_info(data_obj_params=data_obj_params,
                                        TV_params=TV_params,
                                         model_params=model_params, 
                                         fit_params=fit_params)
    else:
        func_info = BN_load_raw_func_info(data_obj_params=data_obj_params)

    return func_info 


# ===================

def BN_load_dn_func_info(data_obj_params, TV_params, model_params, fit_params):
    
    denoiseTV_params = TV_params["denoiseTV_params"]
    denoise_fit_params = fit_params["denoise_fit_params"]
    denoise_model_params = model_params["denoise_model_params"]

    func_info =  BN_load_raw_func_info(data_obj_params=data_obj_params)

    if denoiseTV_params["denoiseVF"]:
        func_info.update(unravel_one_level(denoiseTV_params))
    else:
        func_info.update({"denoiseVF":0})

    func_info.update(unravel_one_level(denoise_model_params))
    func_info.update(denoise_fit_params)

    return func_info


def BN_load_dn_data(data_obj_params, TV_params, model_params, fit_params):
    
    additional_params = data_obj_params["additional_params"]
    dataObj_path = additional_params["denoise_path"]
    denoise_model_params = model_params["denoise_model_params"]
    denoise_fit_params = fit_params["denoise_fit_params"]
    denoiseModelLabel = denoise_fit_params["denoiseModelLabel"]

    func_info = BN_load_dn_func_info(data_obj_params=data_obj_params, 
                                     TV_params=TV_params,
                                     model_params=model_params,
                                     fit_params=fit_params)

    info_path = dictToPath(func_info)
    load_path = os.path.join(os.path.join(dataObj_path, info_path,f"data_dn_num{denoiseModelLabel}.npy"))
    data_obj = np.load(load_path, allow_pickle=True).item(0)

    return func_info, data_obj

# ===================

def BN_load_raw_func_info(data_obj_params):

    RDEq_params_store = data_obj_params["RDEq_params_store"]
    RDEq_params = data_obj_params["RDEq_params"]
    add_noise_params = data_obj_params["add_noise_params"]
    x1 = RDEq_params_store["x1"] 
    x2 = RDEq_params_store["x2"] 
    t = RDEq_params_store["t"] 
    K = RDEq_params_store["K"]
    
    func_info = {
        "dataX1num":len(x1),
        "dataX2num":len(x2),
        "dataTnum":len(t),
        "dataK": K
    }

    func_info.update(RDEq_params)
    func_info.update(add_noise_params) 

    return func_info


def BN_load_raw_data(data_obj_params):

    additional_params = data_obj_params["additional_params"]
    dataObj_path = additional_params["inital_path"]

    func_info = BN_load_raw_func_info(data_obj_params=data_obj_params)
    info_path = dictToPath(func_info)

    load_path = os.path.join(os.path.join(dataObj_path, info_path,"data_obj.npy"))
    #load_path2 = os.path.join(dataObj_path, "dataX1num_12/dataX2num_12/dataTnum_5/dataICLabel_cos/dataDiffLabel_const/dataGrowLabel_const/dataGamma_0/dataNoisePercent_0/dataNoiseSeed_0")

    # print(f"Load path={load_path}")
    # print(f"Load path={load_path2}")

    data_obj = np.load(load_path, allow_pickle=True).item(0)

    return func_info, data_obj



# EXTRAS

def BN_load_raw_data_binn(data_obj_params):

    func_info = BN_load_raw_func_info(data_obj_params)
    info_path = dictToPath(func_info)
    load_path = os.path.join(os.path.join(dataObj_path, info_path,"data_obj.npy"))
    data_obj = np.load(load_path, allow_pickle=True).item(0)

    return func_info, data_obj