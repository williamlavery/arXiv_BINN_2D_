# Model construction 


def DN_model_construction_func_info(model_params):
    denoise_model_params = model_params["denoise_model_params"]
    #func_info = denoise_model_params["denoise_construction_params"] 
    #data_loss_func_info = DN_model_data_loss_func_info(model_params)
    #func_info.update(data_loss_func_info)
    return unravel_one_level(denoise_model_params)


def DN_model_construction(data_obj, 
                          model_params):
    denoise_model_params = model_params["denoise_model_params"]
    denoise_construction_params = denoise_model_params["denoise_construction_params"] 
    denoiseUsize = denoise_construction_params["denoiseUsize"]
    device =  denoise_construction_params["denoiseDevice"]

    _, data_loss_func = DN_model_data_loss_func(model_params=model_params)        

    NN_dn = Denoise(data_obj=data_obj,
                    model_params=model_params,
                    data_loss_func=data_loss_func)

    func_info = DN_model_construction_func_info(model_params=model_params)

    return  func_info, NN_dn