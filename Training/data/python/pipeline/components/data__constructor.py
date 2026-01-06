
def DATA_data_construct(data_obj_params, u, u_clean, theta_D, theta_G):
    
    RDEq_params_store = data_obj_params["RDEq_params_store"]
    add_noise_params =  data_obj_params["add_noise_params"]

    x1 = RDEq_params_store["x1"]
    x2 = RDEq_params_store["x2"]
    t = RDEq_params_store["t"]
    K = RDEq_params_store["K"]


    gamma       = add_noise_params["dataGamma"]

    data_obj = Data(x1= x1,
                    x2 = x2,
                    t = t,
                    u_clean = u_clean,
                    u = u,
                    gamma = gamma,
                    theta_D = theta_D,
                    theta_G = theta_G,
                    K=K)

    return data_obj