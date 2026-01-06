# DONE


def BN_model_construction_func_info(model_params):
    return unravel_one_level(binn_model_params)

def BN_params_binn_builder(data_obj, data_obj_params):

    diffLabel = data_obj_params["RDEq_params"]["dataDiffLabel"] 
    growLabel = data_obj_params["RDEq_params"]["dataGrowLabel"] 

    # ============= diffusion functions ========================
    if diffLabel =="const":
        #theta_D = [0.02472]  # diffusion parameters
        def diffusion_func(u):
            return diffusion_func1(u, data_obj.theta_D)

    if diffLabel =="linear":
        #theta_D = [0.015, 0.06]  # diffusion parameters
        def diffusion_func(u):
            return diffusion_func2(u, data_obj.theta_D)
        
    if diffLabel =="quadratic":
        #theta_D = [0.01, 0.044]  # diffusion parameters
        def diffusion_func(u):
            return diffusion_func3(u, data_obj.theta_D)

    if diffLabel =="exp":
        #theta_D = [0.003, 0.095, 2.5]  # diffusion parameters
        def diffusion_func(u):
            return diffusion_func4(u, data_obj.theta_D) 
    
    # ============= growth functions ========================
    if growLabel =="const":
       # theta_G = [1.3]
        def growth_func(u):
            return growth_func1(u, data_obj.theta_G)

    if growLabel =="linear":
       # theta_G = [2.4,-3]
        def growth_func(u):
            return growth_func2(u, data_obj.theta_G)
        
    if growLabel =="quadratic":
       # theta_G = [2.1,-0.29]
        def growth_func(u):
            return growth_func3(u, data_obj.theta_G)

    if growLabel =="exp":
       # theta_G = [0.7, 1.3, -4]  
        def growth_func(u):
            return growth_func4(u, data_obj.theta_G)
    if growLabel =="zero":
       # theta_G = [0]
        def growth_func(u):
            return growth_func1(u, data_obj.theta_G)

    RDEq_extra_params = {}
    RDEq_extra_params["thetaD"] =data_obj.theta_D
    RDEq_extra_params["thetaG"] =data_obj.theta_G
    RDEq_extra_params["diffusionTrueFunc"] =diffusion_func
    RDEq_extra_params["growthTrueFunc"] =growth_func
    data_obj_params["RDEq_params_store"]["u_clean"] = data_obj.u_clean
    data_obj_params["RDEq_extra_params"] = RDEq_extra_params

    return data_obj_params

def BN_model_construction(data_obj_orig, 
                         data_obj_params, 
                        model_params):
    binn_model_params = model_params["binn_model_params"]
    binn_construction_params = binn_model_params["binn_construction_params"]
    binnUsize = binn_construction_params["binnUsize"]
    binnDsize = binn_construction_params["binnDsize"]
    D_one_param_bool = binn_construction_params["DoneParamBool"]
    binnGsize = binn_construction_params["binnGsize"]
    device =  binn_construction_params["binnDevice"]

    data_obj_params = BN_params_binn_builder(data_obj=data_obj_orig,
                                    data_obj_params=data_obj_params)

    data_obj_params["RDEq_extra_params"]["max_u_clean"] = data_obj_orig.u_clean.max()
    data_obj_params["RDEq_extra_params"]["min_u_clean"] = data_obj_orig.u_clean.min()


    _, data_loss_func = BN_model_data_loss_func(model_params=model_params)                               
    _, pde_loss_func = BN_model_pde_loss_func(model_params=model_params)    


    NN_binn = BINN_2d(data_obj_params=data_obj_params, 
                   model_params=model_params,
                   data_loss_func=data_loss_func,
                   pde_loss_func=pde_loss_func)

    func_info = BN_model_construction_func_info(model_params=model_params)

    return  func_info, NN_binn

