def DN_model_finder(path_intro, data_obj_params, TV_params, model_params, fit_params, mkdir=True):

    all_func_info = {}
    func_info_inital = DN_load_raw_func_info(data_obj_params=data_obj_params)
    all_func_info.update(func_info_inital)
    func_info_TV = DN_TVsplit_func_info(TV_params=TV_params)
    all_func_info.update(func_info_TV)
    func_info_build = DN_model_construction_func_info(model_params=model_params)
    all_func_info.update(func_info_build )
    func_info_fit = DN_model_fit_func_info(fit_params=fit_params)
    all_func_info.update(func_info_fit)

    full_path = os.path.join(path_intro, dictToPath(all_func_info))

    if os.path.exists(full_path) and os.listdir(full_path):
        #print(f"⚠️  Model directory already exists and is non-empty: {full_path}")
        return full_path, 1
    else:
        #print(f"✅ New model will be saved to: {full_path}")
        if mkdir:
            os.makedirs(full_path, exist_ok=True)
        return full_path, 0


def DN_model_fit_again(model_dir_path, model_label, fit_params):



    model_path_full = os.path.join(model_dir_path, f"denoiseModel{model_label}.pth")
    denoise_fit_params = fit_params["denoise_fit_params"]
    denoise_fit_params_additionals = fit_params["denoise_fit_params_additionals"]
    denoiseEpochs = denoise_fit_params_additionals["denoiseEpochs"]

    modelW = torch.load(model_path_full)#, map_location=torch.device(device))
    total_train_losses = len(modelW.train_loss_list)
    print("Currently have trained {} epochs\n".format(total_train_losses))

    if modelW.manual_termination is not None and total_train_losses >= modelW.manual_termination:
        print("\n Stopped Training. Manual termination. Epochs = ", total_train_losses)
        return modelW
    if modelW.early_stopping is not None and total_train_losses - modelW.last_improved >=  modelW.early_stopping:
        print("\n Stopped Training. Early stopping.. Epochs = ", total_train_losses)
        return modelW

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

    return modelW

def DN_retrain_dn(model_dir_path,model_label,denoiseTVsplitSeed,fit_params):

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f"++++++++++++++++++++++++++++++  DENOISE RETRAINING (model_num ={model_label}, denoiseTVsplitSeed = {denoiseTVsplitSeed}) ++++++++++++++++++++++++++++++")
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # =================================== Fit the model =====================================
    model = DN_model_fit_again(model_dir_path=model_dir_path, 
                              model_label=model_label,
                              fit_params=fit_params)

    return model