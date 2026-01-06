# DONE


def BN_model_initalize_func_info(model_params):
    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]
    binnInitializeDenoiseBool = binn_construction_params["binnInitializeDenoiseBool"]
    
    func_info = {"binnInitializeDenoiseBool": binnInitializeDenoiseBool}
    return func_info


def BN_model_initalize(path_intro, NN_binn, model_params, fit_params):

    dn_file_path,_ = DN_model_finder(path_intro=denoise_path,
                                         data_obj_params=data_obj_params,
                                        TV_params =TV_params, 
                                        model_params=model_params,
                                        fit_params=fit_params,
                                        mkdir=False)


    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]
    denoise_model_params = model_params["denoise_model_params"]
    denoise_fit_params_additionals = fit_params["denoise_fit_params_additionals"]
    combineDenoise = denoise_fit_params_additionals["combineDenoise"]
    
    if binn_construction_params["binnInitializeDenoiseBool"]:
        denoise_fit_params = fit_params["denoise_fit_params"]
        denoiseModelLabel = denoise_fit_params["denoiseModelLabel"]

        #for name, param in NN_binn.surface_fitter.named_parameters():
            #print(f"PRE— {name}: {param.view(-1)[:5]}")
          #  break  # Remove this if you want to print more layers
        if not combineDenoise:
            denoise_path_full = os.path.join(dn_file_path, f"denoiseModel{denoiseModelLabel}.pth")
            denoise_loaded = torch.load(denoise_path_full, weights_only=False)#map_location=device)
            
            if denoise_fit_params["denoiseManualTermination"] is None: 
                denoise_loaded.load_best_val() # Load terminated weights
            else:
                denoise_loaded.load_best_terminated() # Load best validation weights


            surface_fitter_state_dict = denoise_loaded.model.surface_fitter.state_dict()
       
        # Load in weights from combining denoiser models

        else:
            TV_params["denoiseTV_params"]["denoiseTVsplitSeed"] = combineDenoise

            dn_file_path,_ = DN_model_finder(path_intro=denoise_path,
                                                    data_obj_params=data_obj_params,
                                                    TV_params =TV_params, 
                                                    model_params=model_params,
                                                    fit_params=fit_params,
                                                    mkdir=False)
                                                    
            denoise_path_full = os.path.join(dn_file_path, f"denoiseModelEnsemble{denoiseModelLabel}.pth")
            denoise_loaded = torch.load(denoise_path_full, weights_only=False)#map_location=device)
            surface_fitter_state_dict = denoise_loaded.state_dict()
        
        NN_binn.surface_fitter.load_state_dict(surface_fitter_state_dict)

        #for name, param in  NN_binn.surface_fitter.named_parameters():
           #print(f"LOADED — {name}: {param.view(-1)[:5]}")
            #break  # Make sure it's the same layer

    func_info = BN_model_initalize_func_info(model_params=model_params)

    return  func_info, NN_binn

