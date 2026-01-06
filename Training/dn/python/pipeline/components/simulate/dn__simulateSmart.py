def DN_sim_smart(data_obj_params,TV_params, model_params, fit_params):

    denoise_model_params = model_params["denoise_model_params"]
    additional_params = data_obj_params["additional_params"]
    denoiseTV_params = TV_params["denoiseTV_params"]
    denoise_fit_params = fit_params["denoise_fit_params"]

    # First check for model existence
    denoise_construction_params = denoise_model_params["denoise_construction_params"]
    denoiseModelLabel = denoise_fit_params["denoiseModelLabel"]

    device = denoise_construction_params["denoiseDevice"]

    denoise_path = additional_params["denoise_path"]

    denoiseTVsplitSeed = denoiseTV_params["denoiseGenerateIndicesArgs"]["denoiseTVsplitSeed"]

    model_save_dir, exist_bool = DN_model_finder(path_intro=denoise_path, 
                                                 data_obj_params=data_obj_params, 
                                                 TV_params=TV_params,
                                                 model_params= model_params,
                                                 fit_params=fit_params)

    model_save_path = os.path.join(model_save_dir, f"denoiseModel{denoiseModelLabel}.pth")
    
    if os.path.isfile(model_save_path):
        modelW = DN_retrain_dn(model_dir_path=model_save_dir, 
                                    model_label=denoiseModelLabel,
                                    denoiseTVsplitSeed=denoiseTVsplitSeed,
                                    fit_params=fit_params)

        _, data_obj = DN_load_raw_data(data_obj_params=data_obj_params)

    else:
        data_obj, modelW = DN_first_train_smart(model_save_dir=model_save_dir,
                    data_obj_params=data_obj_params,
                    TV_params=TV_params,
                    model_params=model_params,
                    fit_params=fit_params)

    DN_save_model(modelW=modelW, 
                 model_save_dir=model_save_dir,
                data_obj=data_obj,
                data_obj_params=data_obj_params,
                TV_params=TV_params,
                model_params=model_params,
                fit_params=fit_params)

    