# ============================ Load the data in to processs  ============================
def DN_load_raw_func_info(data_obj_params):

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


def DN_load_raw_data(data_obj_params):

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

