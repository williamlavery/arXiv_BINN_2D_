def BN_model_finder(path_intro, data_obj_params, TV_params, model_params, fit_params):

    all_func_info = {}
    func_info_inital = BN_load_func_info(data_obj_params=data_obj_params, 
                                        TV_params=TV_params,
                                        model_params=model_params, 
                                        fit_params=fit_params)
    all_func_info.update(func_info_inital)

    func_info_TV = BN_TVsplit_func_info(TV_params=TV_params)
    all_func_info.update(func_info_TV)

    func_info_build = BN_model_construction_func_info(model_params=model_params)
    all_func_info.update(func_info_build )

    func_info_init = BN_model_initalize_func_info(model_params=model_params)
    all_func_info.update(func_info_init)

    func_info_fit = BN_model_fit_smart_func_info(fit_params=fit_params)
    all_func_info.update(func_info_fit)

    full_path = os.path.join(path_intro, dictToPath(all_func_info))

    if os.path.exists(full_path) and os.listdir(full_path):
        #print(f"⚠️  Model directory already exists and is non-empty: {full_path}")
        return full_path, 1
    else:
        #print(f"✅ New model will be saved to: {full_path}")
        os.makedirs(full_path, exist_ok=True)
        return full_path, 0


def BN_model_check_ES(path_intro, data_obj_params, TV_params, model_params, fit_params):

    import shutil

    def update_path(path, ES_old, ES_new):
        """
        Replace 'combineDenoise_[0,1,2]' with 'combineDenoise_None' in a path string.
        Works without touching the filesystem.
        """
        return path.replace(f"binnES_{ES_old}", f"binnES_{ES_new}")

    def copy_dir_contents(src, dest):
        """
        Copies all contents (files & subdirectories) from src to dest.
        If dest doesn't exist, it will be created.
        """
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source directory does not exist: {src}")
        
        os.makedirs(dest, exist_ok=True)

        for item in os.listdir(src):
            src_path = os.path.join(src, item)
            dest_path = os.path.join(dest, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dest_path)

    model_save_dir_current, exist_bool = BN_model_finder(
                        path_intro=path_intro, 
                        data_obj_params=data_obj_params, 
                        TV_params = TV_params,
                        model_params= model_params, 
                        fit_params=fit_params)
    
    if exist_bool:
        return 

    fit_params_to_update = fit_params.copy()
    ES_current = fit_params["binn_fit_params"]["binnES"]

    sorted_ES = np.array(sorted(fit_params["binn_fit_params_additionals"]["binnES_check"], reverse=True))


    for ES in sorted_ES[sorted_ES< ES_current]:
        fit_params_to_update["binn_fit_params"]["binnES"] = ES

        # Precisely the same as model_save_dir_current, but with the a lower ES
        model_save_dir_old, exist_bool = BN_model_finder(
                        path_intro=path_intro, 
                        data_obj_params=data_obj_params, 
                        TV_params = TV_params,
                        model_params= model_params, 
                        fit_params=fit_params_to_update)
        
        if exist_bool:

            binnModelLabel = fit_params["binn_fit_params"]["binnModelLabel"]
            copy_dir_contents(model_save_dir_old, model_save_dir_current)

            model_path = os.path.join(model_save_dir_current, f"binnModel{binnModelLabel}.pth")


            modelW = torch.load(model_path, weights_only=False)


            modelW.early_stopping = ES_current
            modelW.save_name = update_path(modelW.save_name, ES_old = ES, ES_new = ES_current)
            torch.save(modelW, model_path)

            print("-----------------------------------------------")
            print(f"Utilised model with ES={ES}")
            print("-----------------------------------------------")

            return 1

def BN_model_fit_again(model_dir_path, TV_params, fit_params,load_ES_bool=True):

    binnModelLabel = fit_params["binn_fit_params"]["binnModelLabel"]
    model_path_full = os.path.join(model_dir_path, f"binnModel{binnModelLabel}.pth")
    binn_fit_params = fit_params["binn_fit_params"]
    binn_fit_params_additionals = fit_params["binn_fit_params_additionals"]
    binnEpochs = binn_fit_params_additionals["binnEpochs"]
            
    modelW = torch.load(model_path_full, weights_only=False)# map_location=torch.device(device))
    total_train_losses = len(modelW.train_loss_list)
    print("Currently have trained {} epochs\n".format(total_train_losses))


   # if modelW.manual_termination is not None and total_train_losses >= modelW.manual_termination:
    #    print("\n Stopped Training. Manual termination. Epochs = ", total_train_losses)
     #   return modelW
    if modelW.early_stopping is not None and total_train_losses - modelW.last_improved >=  modelW.early_stopping:
        print("\n Stopped Training. Early stopping.. Epochs = ", total_train_losses)
        return modelW
    
    if load_ES_bool:
        modelW.load_ES()
    else:
        modelW.load_expired()

    modelW.fit(
                x_tr_input=modelW.x_train_torch,
                y_tr_input=modelW.y_train_torch,
                batch_size=modelW.batch_size,
                epochs=int(binnEpochs),
                verbose=modelW.verbose,
                validation_data=modelW.validation_data,
                early_stopping=modelW.early_stopping,
                rel_update_thresh = modelW.rel_update_thresh,
                rel_save_thresh=modelW.rel_save_thresh)

    return modelW

def BN_retrain(model_dir_path, TV_params, fit_params, load_ES_bool=True):

    binnModelLabel = fit_params["binn_fit_params"]["binnModelLabel"]
    binnSplitSeed = TV_params["binnTV_params"]["binnGenerateIndicesArgs"]["binnTVsplitSeed"]
    
    print(f"--------------------------------------------------------------------------------------")
    print(f"-------------------------- BINN RETRAINING (model_num ={binnModelLabel}, binnSplitSeed = {binnSplitSeed}) ---------------------------")
    print(f"--------------------------------------------------------------------------------------")

    # =================================== Fit the model =====================================
    model = BN_model_fit_again(model_dir_path=model_dir_path, TV_params=TV_params,fit_params=fit_params, load_ES_bool=load_ES_bool)

    return model