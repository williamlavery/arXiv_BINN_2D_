# DONE


# ================ data loss ==========================
def DN_model_data_loss_func_info(model_params):
    denoise_model_params = model_params["denoise_model_params"]
    func_info = denoise_model_params["DNdata_loss_params"]
    return func_info

def DN_model_data_loss_func(model_params):
    denoise_model_params = model_params["denoise_model_params"]
    DNdata_loss_params = denoise_model_params["DNdata_loss_params"]
    DNdataLossFuncLabel = DNdata_loss_params["DNdataLossFuncLabel"]
    
    if DNdataLossFuncLabel == "MSE":
        DN_model_data_loss_func = data_loss_MSE
    if DNdataLossFuncLabel == "GLS":
        DN_model_data_loss_func = data_loss_GLS
    func_info =  DN_model_data_loss_func_info(model_params)
    return func_info, DN_model_data_loss_func



