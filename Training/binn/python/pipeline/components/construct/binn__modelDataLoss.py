# DONE


# ================ data loss ==========================
def BN_model_data_loss_func_info(model_params):
    binn_model_params = model_params["binn_model_params"]
    func_info = binn_model_params["BNdata_loss_params"]
    return func_info


def BN_model_data_loss_func(model_params):

    binn_model_params = model_params["binn_model_params"]
    BNdata_loss_params = binn_model_params["BNdata_loss_params"]
    BNdataLossFuncLabel = BNdata_loss_params["BNdataLossFuncLabel"]
    
    if BNdataLossFuncLabel == "MSE":
        BN_model_data_loss_func = data_loss_MSE
    if BNdataLossFuncLabel == "GLS":
        BN_model_data_loss_func = data_loss_GLS
    func_info =  BN_model_data_loss_func_info(model_params)
    return func_info, BN_model_data_loss_func





