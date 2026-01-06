#  ==========================  Binn first training  ==========================




# Function information 
# =====================
def BN_model_fit_func_info(fit_params):
    func_info = fit_params["binn_fit_params"]
    return  fit_params["binn_fit_params"]

# ==================   

def BN_model_fit_first(model_save_dir, NN_binn, fit_params, TV_dic):

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

    binnLR = binn_fit_params["binnLR"]
    binnBatchSize = binn_fit_params["binnBatchSize"]
    binnRelUpdateThresh = binn_fit_params["binnRelUpdateThresh"]
    binnRelSaveThresh = binn_fit_params["binnRelSaveThresh"]
    binnES = binn_fit_params["binnES"]
    binnEpochs = binn_fit_params["binnEpochs"]

    parameters = NN_binn.parameters()
    opt = torch.optim.Adam(parameters, lr=binnLR)

    weights_path_relative = os.path.join(model_save_dir,f'Weights_binn_num{seed}')
    os.makedirs(weights_path_relative, exist_ok=True)

    modelW = ModelWrapper(
        model=NN_binn ,
        optimizer=opt,
        loss=NN_binn.loss,
        save_name='{}/test'.format(weights_path_relative))

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
                x_tr=modelW.x_train_torch,
                y_tr=modelW.y_train_torch,
                batch_size=modelW.batch_size,
                epochs=int(binnEpochs),
                verbose=modelW.verbose,
                validation_data=modelW.validation_data,
                early_stopping=modelW.early_stopping,
                rel_update_thresh = modelW.rel_update_thresh,
                rel_save_thresh=modelW.rel_save_thresh)

    func_info = BN_model_fit_func_info(fit_params=fit_params)
 
    return  func_info,modelW





def BN_first_train(model_save_dir,
                    data_obj_params,
                    model_params,
                    fit_params):
    
    binn_fit_params = fit_params["binn_fit_params"]
    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]

    if binn_fit_params["twoStepBool"]:

        _, data_obj_orig = BN_load_raw_data(data_obj_params)
        _, data_obj = BN_load_dn_data(data_obj_params, model_params, fit_params)

    
    else:

         # =================================== Load data =====================================
        _, data_obj_orig = BN_load_raw_data(data_obj_params)

        data_obj = data_obj_orig
   
    # =================================== Split the data  =====================================
    _, TV_dic = BN_TVsplit(data_obj=data_obj, model_params = model_params, fit_params=fit_params)
    
    # =================================== construct model  =====================================
    _, NN_binn = BN_model_construction(data_obj_orig=data_obj_orig,
                                    data_obj_params=data_obj_params,
                                    model_params=model_params,
                                    )
    # ================================== initalize as denoise =====================================
    
    if binn_construction_params["binnInitializeDenoiseBool"]:
        dnPath = data_obj_params["denoisePath"]
        dn_file_path,_ = DN_model_finder(dnPath, data_obj_params, model_params, fit_params)
        _, NN_binn = BN_model_initalize(NN_binn, model_params, dn_file_path)

    binnModelLabel = binn_construction_params["binnModelLabel"]
    binn_TV_additionals = binnTV_params["binn_generate_indices_args"]

    print(f"--------------------------------------------------------------------------------------")
    print(f"-------------------------- BINN FIRST TRAINING (model_num ={binnModelLabel}, binn_TV_additionals = {binn_TV_additionals}) ---------------------------")
    print(f"--------------------------------------------------------------------------------------")


    # =================================== Fit the model =====================================
    func_info, modelW = BN_model_fit_first(model_save_dir=model_save_dir, NN_binn=NN_binn, fit_params=fit_params, TV_dic=TV_dic)

    return data_obj, modelW