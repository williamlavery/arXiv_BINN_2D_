# Model fit


# Function information 
# =====================
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
    device = model_params["denoiseDevice"]


    x_train = to_torch_grad(x_train_np, device)
    y_train = to_torch_grad(y_train_np, device)

    if validation_data:
        x_val = to_torch_grad(x_val_np, device)
        y_val = to_torch_grad(y_val_np, device)
        validation_data = [x_val,y_val]

    denoise_fit_params = fit_params["denoise_fit_params"]
    denoiseLR = fit_params["denoiseLR"]
    denoiseBatchSize = fit_params["denoiseBatchSize"]
    denoiseRelUpdateThresh = fit_params["denoiseRelUpdateThresh"]
    denoiseRelSaveThres = fit_params["denoiseRelSaveThresh"]
    denoiseES = fit_params["denoiseES"]
    denoiseEpochs = fit_params["denoiseEpochs"]

    parameters = NN_dn.parameters()
    opt = torch.optim.Adam(parameters, lr=denoiseLR )


    weights_path_relative = os.path.join(model_save_dir,f'Weights_dn_num{seed}')
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
                rel_update_thresh = modelW.rel_update_thresh,
                rel_save_thresh=modelW.rel_save_thresh)

    func_info = DN_model_fit_func_info(fit_params=fit_params)

    return  func_info,modelW

def DN_first_train(model_save_dir,
                    data_obj_params,
                    model_params,
                    fit_params):

    denoise_construction_params = denoise_model_params["denoise_construction_params"]
    denoiseTV_params = TV_params["denoiseTV_params"]
    
    # =================================== Load data =====================================
    _, data_obj = DN_load_raw_data(data_obj_params=data_obj_params)
   
    # =================================== Split the data  =====================================
    _, TV_dic = DN_TVsplit(data_obj=data_obj, 
                           TV_params=TV_params,
                           model_params = model_params)
    
    # =================================== construct model  =====================================
    _, NN_dn = DN_model_construction(data_obj=data_obj, 
                                     model_params=model_params)

    denoise_TV_additionals = denoiseTV_params["binnGenerateIndicesArgs"]
    denoiseModelLabel = denoise_construction_params["denoiseModelLabel"]
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"++++++++++++++++++++++++++++++ FIRST TRAINING DENOISE(model_num ={denoiseModelLabel}, dn_TV_additionals = {denoise_TV_additionals}) ++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # =================================== Fit the model =====================================
    func_info, modelW = DN_model_fit_first(model_save_dir=model_save_dir, NN_dn=NN_dn, fit_params=fit_params, TV_dic=TV_dic)

    return data_obj, modelW
