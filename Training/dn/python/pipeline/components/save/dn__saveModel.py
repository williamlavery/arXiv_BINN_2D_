#def print_path_components(path):
 #   components = os.path.normpath(path).split(os.sep)
  #  for part in components:
   #     print(part)

def DN_save_model(modelW, 
                  model_save_dir,
                data_obj,
                data_obj_params,
                TV_params, 
                model_params,
                fit_params):


    # =================================== Save the model =====================================
    denoiseModelLabel = fit_params["denoise_fit_params"]["denoiseModelLabel"]
    torch.save(modelW, os.path.join(model_save_dir, f"denoiseModel{denoiseModelLabel}.pth"))


    #print("The inital model training has finished and the model has been saved as: \n", '{} \n'.format(model_save_path_full))
    # update text file

    print('******************************')
    print(f"Model number =", denoiseModelLabel)
    print(f"dn_TV_additionals = ", TV_params["denoiseTV_params"]["denoiseGenerateIndicesArgs"])
    print("Number of trained epochs =", len(modelW.train_loss_list))
    print('******************************')

    # =================================== Form & save corresponding data_obj =====================================
    device = model_params["denoise_model_params"]["denoise_construction_params"]["denoiseDevice"]
    
    DN_save_model_generated_dataObj(model_loaded = modelW,
                                 dataobj=data_obj, 
                                 model_label=denoiseModelLabel, 
                                 path_to_dataObj_dir = model_save_dir,
                                 device=device)

    if fit_params["denoise_fit_params_additionals"]["combineDenoise"]:

      DN_generate_ensemble_dataObj(model_label=denoiseModelLabel,
                                      path_to_dataObj_dir = model_save_dir,
                                      data_obj_params=data_obj_params,
                                        TV_params=TV_params,
                                        model_params=model_params,
                                        fit_params=fit_params)
