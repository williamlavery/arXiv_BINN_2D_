# ==================== pde loss ====================
def BN_model_pde_loss_func_info(model_params):
    binn_model_params=model_params["binn_model_params"]
    func_info = binn_model_params["pde_loss_params"]
    return func_info


def BN_model_pde_loss_func(model_params):

    binn_model_params=model_params["binn_model_params"]
    pde_loss_params = binn_model_params["pde_loss_params"]
    BCbool = pde_loss_params["BCbool"]

    if BCbool:
        pde_loss_func = pde_loss_with_bc_2d
    else:
        pde_loss_func = pde_loss_without_bc_2d

    func_info  = BN_model_pde_loss_func_info(model_params)
    return func_info, pde_loss_func
