def BN_save_model(modelW, model_save_dir, data_obj, TV_params, model_params, fit_params):

    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]

    binn_fit_params = fit_params["binn_fit_params"]
    binnModelLabel = binn_fit_params["binnModelLabel"]
    device =  binn_construction_params["binnDevice"]

    binnTV_params = TV_params["binnTV_params"] 
    binnGenerateIndicesArgs= binnTV_params["binnGenerateIndicesArgs"]

    # =================================== Save the model =====================================
    model_save_path_full = os.path.join(model_save_dir, f"binnModel{binnModelLabel}.pth")
    torch.save(modelW, model_save_path_full)


    #print("The inital model training has finished and the model has been saved as: \n", '{} \n'.format(model_save_path_full))
    # update text file
    print('******************************')
    print(f"Model number =", binnModelLabel)
    print(f"binnGenerateIndicesArgs= ", binnGenerateIndicesArgs)
    print("Number of trained epochs =", len(modelW.train_loss_list))
    print('******************************')

    # =================================== Form & save corresponding data_obj =====================================
    BN_save_model_generated_dataObj(model_loaded = modelW,
                                 dataobj=data_obj, 
                                 model_label=binnModelLabel, 
                                 path_to_dataObj_dir = model_save_dir,
                                 device=device)