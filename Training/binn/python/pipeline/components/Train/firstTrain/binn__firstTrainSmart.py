#  ==========================  Binn first training  ==========================


# Function information 
# =====================
def BN_model_fit_smart_func_info(fit_params):
    return  fit_params["binn_fit_params"]

# ==================   

def BN_model_fit_first_smart(model_save_dir, NN_binn, fit_params, TV_dic):

    train_dic = TV_dic["train_dic"]
    val_dic =  TV_dic["val_dic"]
    validation_data = val_dic["validation_data"]
    x_train_np = train_dic["x_train_np"]
    y_train_np = train_dic["y_train_np"]
    x_val_np = val_dic["x_val_np"]
    y_val_np = val_dic ["y_val_np"]

    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]
    device = binn_construction_params["binnDevice"]


    x_train = to_torch_grad(x_train_np, device)
    y_train = to_torch_grad(y_train_np, device)

    if validation_data:
        x_val = to_torch_grad(x_val_np, device)
        y_val = to_torch_grad(y_val_np, device)
        validation_data = [x_val,y_val]


    binn_fit_params = fit_params["binn_fit_params"]
    binn_fit_params_additionals = fit_params["binn_fit_params_additionals"]
    binnEpochs = binn_fit_params_additionals["binnEpochs"]
    printFreq = binn_fit_params_additionals["printFreq"]
    
    binnLR = binn_fit_params["binnLR"]
    binnBatchSize = binn_fit_params["binnBatchSize"]
    binnRelUpdateThresh = binn_fit_params["binnRelUpdateThresh"]
    binnRelSaveThresh = binn_fit_params["binnRelSaveThresh"]
    binnES = binn_fit_params["binnES"]

    binn_fit_params = fit_params["binn_fit_params"]
    binnModelLabel = binn_fit_params["binnModelLabel"]

    parameters = NN_binn.parameters()
    opt = torch.optim.Adam(parameters, lr=binnLR)
    weights_path_relative = os.path.join(model_save_dir,f'Weights_binn_num{binnModelLabel}')
    os.makedirs(weights_path_relative, exist_ok=True)

    modelW = ModelWrapper(
        model=NN_binn ,
        optimizer=opt,
        loss=NN_binn.loss,
        save_name='{}/test'.format(weights_path_relative))
    
    modelW.model_save_dir = model_save_dir
    modelW.binnModelLabel = binnModelLabel

    modelW.x_train = x_train_np
    modelW.y_train = y_train_np
    modelW.x_val = x_val_np
    modelW.y_val = y_val_np

    modelW.x_train_torch = x_train
    modelW.y_train_torch = y_train
    modelW.validation_data = validation_data

    modelW.batch_size=binnBatchSize
    modelW.early_stopping=binnES
    modelW.rel_update_thresh = binnRelUpdateThresh
    modelW.rel_save_thresh=binnRelSaveThresh

    verbose = 1
    modelW.verbose=verbose

    
    # store training and validation within wrapper for conveience

    # train jointly


    modelW.fit(
                x_tr_input=modelW.x_train_torch,
                y_tr_input=modelW.y_train_torch,
                batch_size=modelW.batch_size,
                epochs=int(binnEpochs),
                verbose=modelW.verbose,
                validation_data=modelW.validation_data,
                early_stopping=modelW.early_stopping,
                rel_update_thresh = modelW.rel_update_thresh,
                rel_save_thresh=modelW.rel_save_thresh,
                print_freq = printFreq)
    


    func_info = BN_model_fit_smart_func_info(fit_params=fit_params)

    return  func_info,modelW

def exc_data_pipeline(data_obj_params):
    #  ===================== Generate data object ===================== 
    cwd = os.getcwd()
    new_wd = "../../../../data/python/pipeline/exec"    # exec -> pip -> py ->
    os.chdir(new_wd)
    Modules_path = findUpwardPath('Modules')  # find data path
    config_path = findUpwardPath('config')  # find data path
    components_path = findUpwardPath('components')  # find data path

    ExecAllWithin(Modules_path)
    ExecAllWithin(components_path)
    ExecAllWithin(config_path)
    DATA_sim(data_obj_params=data_obj_params)
    os.chdir(cwd)
    # ===================================================================
    return 


def exc_dn_smart_pipeline(data_obj_params, TV_params, model_params, fit_params):
    #  ===================== Generate data object ===================== 
    cwd = os.getcwd()
    new_wd = "../../../../dn/python/pipeline/exec"    # exec -> pip -> py ->
    os.chdir(new_wd)
    Modules_path = findUpwardPath('Modules')        # find Modules path
    components_path = findUpwardPath('components')  # find components path
    #config_path = findUpwardPath('config')          # find config path
    ExecAllWithin(Modules_path)
    ExecAllWithin(components_path)
    #ExecAllWithin(config_path)
    DN_sim_smart(data_obj_params=data_obj_params,
                 TV_params =TV_params, 
                model_params=model_params,
                fit_params=fit_params)
    os.chdir(cwd)
    # ===================================================================
    return 


def BN_first_train_smart(model_save_dir,
                    data_obj_params,
                    TV_params,
                    model_params,
                    fit_params):
    
    binn_fit_params = fit_params["binn_fit_params"]
    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]
    additional_params = data_obj_params["additional_params"] 
    denoiseTV_params = TV_params["denoiseTV_params"]

    print(f"twoStepBool: {binn_fit_params['twoStepBool']}, binnInitializeDenoiseBool: {binn_construction_params['binnInitializeDenoiseBool']}")
    if binn_fit_params["twoStepBool"] or binn_construction_params["binnInitializeDenoiseBool"]:



        #  ===================== Generate dn data object ===================== 
        exc_dn_smart_pipeline(data_obj_params=data_obj_params,
                                TV_params =TV_params, 
                                model_params=model_params,
                                fit_params=fit_params)

    

    if binn_fit_params["twoStepBool"]:
        _, data_obj_orig = BN_load_raw_data(data_obj_params=data_obj_params)

        if binn_fit_params["combineDenoise"] and denoiseTV_params["denoiseVF"]:
            
            for denoiseTVsplitSeed in binn_fit_params["combineDenoise"]:

                denoiseTV_params = TV_params["denoiseTV_params"]
                denoiseTV_params["denoiseGenerateIndicesArgs"]["denoiseTVsplitSeed"] = denoiseTVsplitSeed
                TV_params["denoiseTV_params"] = denoiseTV_params

                exc_dn_smart_pipeline(data_obj_params=data_obj_params,
                                TV_params =TV_params, 
                                model_params=model_params,
                                fit_params=fit_params)
                                
            _, data_obj = BN_load_dn_data_ensemble(data_obj_params=data_obj_params,
                                        TV_params= TV_params,
                                        model_params=model_params,
                                        fit_params=fit_params)
        else:
            _, data_obj = BN_load_dn_data(data_obj_params=data_obj_params,
                                      TV_params= TV_params,
                                      model_params=model_params,
                                      fit_params=fit_params)


    else:
        #  ===================== Generate data object ===================== 
        exc_data_pipeline(data_obj_params=data_obj_params)

         # =================================== Load data =====================================
        _, data_obj_orig = BN_load_raw_data(data_obj_params=data_obj_params)

        data_obj = data_obj_orig
   
    
    # =================================== Split the data  =====================================
    _, TV_dic = BN_TVsplit(data_obj=data_obj, 
                            #data_obj_params=data_obj_params,
                           model_params = model_params, 
                           TV_params=TV_params)
    
    # =================================== construct model  =====================================
    _, NN_binn = BN_model_construction(data_obj_orig=data_obj_orig,
                                        data_obj_params=data_obj_params,
                                        model_params=model_params)
    # ================================== initalize as denoise =====================================
    
    if binn_construction_params["binnInitializeDenoiseBool"]:
        
        denoise_path = additional_params["denoise_path"]
        dn_file_path,_ = DN_model_finder(path_intro=denoise_path,
                                         data_obj_params=data_obj_params,
                                        TV_params =TV_params, 
                                        model_params=model_params,
                                        fit_params=fit_params)

        _, NN_binn = BN_model_initalize(path_intro=denoise_path,
                                        NN_binn=NN_binn, 
                                        model_params= model_params, 
                                        fit_params=fit_params)

    binn_fit_params = fit_params["binn_fit_params"]
    binnModelLabel = binn_fit_params["binnModelLabel"]
    binnTV_params = TV_params["binnTV_params"]
    binn_TV_additionals = binnTV_params["binnGenerateIndicesArgs"]

    print(f"--------------------------------------------------------------------------------------")
    print(f"-------------------------- BINN FIRST TRAINING (model_num ={binnModelLabel}, binn_TV_additionals = {binn_TV_additionals}) ---------------------------")
    print(f"--------------------------------------------------------------------------------------")


    # =================================== Fit the model =====================================
    func_info, modelW = BN_model_fit_first_smart(model_save_dir=model_save_dir, NN_binn=NN_binn, fit_params=fit_params, TV_dic=TV_dic)

    return data_obj, modelW
