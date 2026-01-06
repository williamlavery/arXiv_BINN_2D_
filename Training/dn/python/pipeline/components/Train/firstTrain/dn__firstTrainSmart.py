# Model fit


def DN_model_fit_func_info(fit_params):
    return  fit_params["denoise_fit_params"]



def DN_model_fit_first(model_save_dir, NN_dn, fit_params, TV_dic):

    train_dic = TV_dic["train_dic"]
    val_dic =  TV_dic["val_dic"]
    validation_data = val_dic["validation_data"]
    x_train_np = train_dic["x_train_np"]
    y_train_np = train_dic["y_train_np"]
    x_val_np = val_dic["x_val_np"]
    y_val_np = val_dic ["y_val_np"]
    u_clean_np = train_dic["u_clean_np"]

    denoise_model_params = model_params["denoise_model_params"]
    denoise_construction_params = denoise_model_params["denoise_construction_params"]
    device = denoise_construction_params["denoiseDevice"]

    denoise_fit_params = fit_params["denoise_fit_params"]
    denoise_fit_params_additionals = fit_params["denoise_fit_params_additionals"]
    denoiseEpochs = denoise_fit_params_additionals["denoiseEpochs"]

    u_clean_torch = torch.tensor(u_clean_np, dtype=torch.float32, device=device)


    x_train = to_torch_grad(x_train_np, device)
    y_train = to_torch_grad(y_train_np, device)

    if validation_data:
        x_val = to_torch_grad(x_val_np, device)
        y_val = to_torch_grad(y_val_np, device)
        validation_data = [x_val,y_val]


    denoiseLR = denoise_fit_params["denoiseLR"]
    denoiseBatchSize = denoise_fit_params["denoiseBatchSize"]
    denoiseRelUpdateThresh = denoise_fit_params["denoiseRelUpdateThresh"]
    denoiseRelSaveThres = denoise_fit_params["denoiseRelSaveThresh"]
    denoiseES = denoise_fit_params["denoiseES"]
    denoiseModelLabel = denoise_fit_params["denoiseModelLabel"]
    denoiseManualTermination = denoise_fit_params["denoiseManualTermination"]

    parameters = NN_dn.parameters()
    opt = torch.optim.Adam(parameters, lr=denoiseLR )
    weights_path_relative = os.path.join(model_save_dir,f'Weights_denoise_num{denoiseModelLabel}')
    os.makedirs(weights_path_relative, exist_ok=True)


    modelW = ModelWrapper_dn(
        model=NN_dn ,
        optimizer=opt,
        loss=NN_dn.loss,
        augmentation=None,
        save_name='{}/test'.format(weights_path_relative))

    modelW.x_train = x_train_np
    modelW.y_train = y_train_np
    modelW.x_val = x_val_np
    modelW.y_val = y_val_np

    modelW.x_train_torch = x_train
    modelW.y_train_torch = y_train
    modelW.validation_data = validation_data

    modelW.batch_size=denoiseBatchSize
    modelW.early_stopping=denoiseES
    modelW.rel_update_thresh = denoiseRelUpdateThresh
    modelW.rel_save_thresh=denoiseRelSaveThresh

    modelW.manual_termination = denoiseManualTermination

    modelW.u_clean_torch_flat = u_clean_torch.flatten()

    verbose = 1
    modelW.verbose=verbose

    modelW.fit(
                x_tr=modelW.x_train_torch,
                y_tr=modelW.y_train_torch,
                batch_size=modelW.batch_size,
                epochs=int(denoiseEpochs),
                verbose=modelW.verbose,
                validation_data=modelW.validation_data,
                early_stopping=modelW.early_stopping,
                manual_termination = modelW.manual_termination,
                rel_update_thresh = modelW.rel_update_thresh,
                rel_save_thresh=modelW.rel_save_thresh)

    func_info = DN_model_fit_func_info(fit_params=fit_params)

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
    DATA_sim(data_obj_params)
    os.chdir(cwd)
    # ===================================================================
    return 

def DN_first_train_smart(model_save_dir,
                        data_obj_params,
                        model_params,
                        TV_params,
                        fit_params):
    denoise_construction_params = denoise_model_params["denoise_construction_params"]
    denoiseTV_params = TV_params["denoiseTV_params"]

    #  ===================== Generate data object ===================== 
    exc_data_pipeline(data_obj_params)
    
    # =================================== Load data =====================================
    _, data_obj = DN_load_raw_data(data_obj_params=data_obj_params)
   
    # =================================== Split the data  =====================================
    _, TV_dic = DN_TVsplit(data_obj=data_obj, 
                           TV_params=TV_params,
                           model_params = model_params)
    
    # =================================== construct model  =====================================

    _, NN_dn = DN_model_construction(data_obj=data_obj, model_params=model_params)


    denoiseGenerateIndicesArgs = denoiseTV_params["denoiseGenerateIndicesArgs"]
    denoise_fit_params = fit_params["denoise_fit_params"]
    denoiseModelLabel = denoise_fit_params["denoiseModelLabel"]

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"++++++++++++++++++++++++++++++  DENOISE FIRST TRAINING(model_num ={denoiseModelLabel}, denoiseGenerateIndicesArgs = {denoiseGenerateIndicesArgs}) ++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    # =================================== Fit the model =====================================
    func_info, modelW = DN_model_fit_first(model_save_dir=model_save_dir, NN_dn=NN_dn, fit_params=fit_params, TV_dic=TV_dic)

    return data_obj, modelW
