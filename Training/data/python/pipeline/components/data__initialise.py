def DATA_RDEq2_func_info(data_obj_params):
    RDEq_params_store = data_obj_params["RDEq_params_store"]
    RDEq_params = data_obj_params["RDEq_params"]
    x1 = RDEq_params_store["x1"] 
    x2 = RDEq_params_store["x2"] 
    t = RDEq_params_store["t"] 
    K = RDEq_params_store["K"]


    return  RDEq_params


def DATA_RDEq2(data_obj_params):
    import re
    print("=============================================================== ")
    print(" ===================== Forward solving for data =============== ")
    print("=============================================================== ")

    x1 = RDEq_params_store["x1"] 
    x2 = RDEq_params_store["x2"] 
    t = RDEq_params_store["t"]
    K = RDEq_params_store["K"] 

    RDEq_params = data_obj_params["RDEq_params"]

    IC_label= RDEq_params["dataICLabel"]
    dataDiffLabel   = data_obj_params["RDEq_params"]["dataDiffLabel"]
    dataGrowLabel  = data_obj_params["RDEq_params"]["dataGrowLabel"]

    clear = False

    if IC_label == "cos":
        def initial_condition(x1, x2):
            return ic1(x1, x2)


    # Example: IC_label = "cosFlat0.5"

    match = re.fullmatch(r"cosFlat(\d*\.?\d+)", IC_label)
    if match:
        amplitude = float(match.group(1))
        def initial_condition(x1, x2):
            return ic1(x1, x2, amplitude=amplitude)


    if IC_label == "scratch":
        def initial_condition(x1, x2):
            return scratch(x1, x2)

        
    # ============= diffusion functions ========================
    if dataDiffLabel =="const":
        theta_D = [0.02472]  # diffusion parameters
        def diffusion_func(u):
            return diffusion_func1(u, theta_D)

    if dataDiffLabel =="linear":
        theta_D = [0.015, 0.06]  # diffusion parameters
        def diffusion_func(u):
            return diffusion_func2(u, theta_D)
        
    if dataDiffLabel =="quadratic":
        theta_D = [0.01, 0.044]  # diffusion parameters
        def diffusion_func(u):
            return diffusion_func3(u, theta_D)

    if dataDiffLabel =="exp":
        theta_D = [0.003, 0.095, 2.5]  # diffusion parameters
        def diffusion_func(u):
            return diffusion_func4(u, theta_D) 
    
    # ============= growth functions ========================
    if dataGrowLabel =="const":
        theta_G = [1.3]
        def growth_func(u):
            return growth_func1(u, theta_G)

    if dataGrowLabel =="linear":
        theta_G = [2.4,-3]
        def growth_func(u):
            return growth_func2(u, theta_G)
        
    if dataGrowLabel =="quadratic":
        theta_G = [2.1,-0.29]
        def growth_func(u):
            return growth_func3(u, theta_G)

    if dataGrowLabel =="exp":
        theta_G = [0.7, -1.3, 4]  
        def growth_func(u):
            return growth_func4(u, theta_G)
    if dataGrowLabel =="zero":
        theta_G = [0]
        def growth_func(u):
            return growth_func1(u, theta_G)

    
    #u0 = initial_condition(x1,x2)
    func_info = DATA_RDEq2_func_info(data_obj_params)
    u_clean = PDE_sim_old_2d_upd(PDE_RHS_2D, initial_condition, x1, x2, t, diffusion_func, growth_func, clear=clear)
    return func_info, u_clean, (theta_D,theta_G)

