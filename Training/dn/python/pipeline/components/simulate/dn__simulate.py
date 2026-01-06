def DN_sim(data_obj_params,model_params, fit_params):

    # First check for model existence
    denoiseModelLabel = model_params["denoiseModelLabel"]
    dnPath = data_obj_params["denoisePath"]
    device = model_params["denoiseDevice"]

    model_save_dir, exist_bool = DN_model_finder(path_intro=dnPath, 
                             data_obj_params=data_obj_params, 
                            model_params= model_params,
                            fit_params=fit_params)
    model_save_path = os.path.join(model_save_dir, f"denoiseModel{denoiseModelLabel}.pth")

    if os.path.isfile(model_save_path):
        modelW = DN_retrain_dn(model_dir_path=model_save_dir, model_label=denoiseModelLabel)
        _, data_obj = DN_load_raw_data(data_obj_params)

    else:
        data_obj, modelW = DN_first_train(model_save_dir=model_save_dir,
                    data_obj_params=data_obj_params,
                    model_params=model_params,
                    fit_params=fit_params)
    DN_save_model(modelW=modelW, model_save_dir=model_save_dir, data_obj=data_obj)