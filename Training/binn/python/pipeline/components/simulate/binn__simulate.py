def BN_sim(data_obj_params, model_params, fit_params):

    binn_model_params = model_params["binn_model_params"]
    additional_params = data_obj_params["additional_params"]

    # First check for model existence
    binnModelLabel = binn_model_params["binnModelLabel"]
    binn_path = additional_params["binn_path"]
    device = binn_model_params["binnDevice"]

    model_save_dir, exist_bool = BN_model_finder(
                            path_intro=binn_path, 
                            data_obj_params=data_obj_params, 
                            model_params= model_params,
                            fit_params=fit_params)
    model_save_path = os.path.join(model_save_dir, f"binnModel{binnModelLabel}.pth")

    if os.path.isfile(model_save_path):
        modelW = BN_retrain(model_dir_path=model_save_dir, model_label=binnModelLabel)
        _, data_obj = BN_load_raw_data(data_obj_params)

    else:

        data_obj, modelW = BN_first_train(
                    model_save_dir=model_save_dir,
                    data_obj_params=data_obj_params,
                    model_params=model_params,
                    fit_params=fit_params)
        
        
    # which ever data_obj we have (original or denoised) the parameters needed in 
    # save_model are the same
    BN_save_model(modelW=modelW, model_save_dir=model_save_dir, data_obj=data_obj)