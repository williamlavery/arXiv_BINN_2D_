

def DATA_finder(data_obj_params):

    all_func_info = {}
    func_info1= DATA_RDEq2_func_info(data_obj_params=data_obj_params)
    all_func_info.update(func_info1)

    func_info2 = DATA_add_noise_func_info(data_obj_params=data_obj_params)
    all_func_info.update(func_info2)

    additional_params = data_obj_params["additional_params"]
    data_obj_path = additional_params["inital_path"]
    full_path = os.path.join(data_obj_path, dictToPath(all_func_info))

    if os.path.exists(full_path) and os.listdir(full_path):
        #print(f"⚠️  Data directory already exists and is non-empty: {full_path}")
        return full_path, 1
    else:
        #print(f"✅ Data will be saved to: {full_path}")
        os.makedirs(full_path, exist_ok=True)
        return full_path, 0



def DATA_sim(data_obj_params):

    data_path_dir, file_present = DATA_finder(data_obj_params)
    data_path_file = os.path.join(data_path_dir,"data_obj.npy")

    additional_params = data_obj_params["additional_params"]
    
    if file_present and not additional_params["overwrite_bool"]:
        print("No new data generated")
    
    else:

        # ========================== Initalize ==========================
        _, u_clean, (theta_D, theta_G) = DATA_RDEq2(data_obj_params)

        # ========================== add Noise ==========================
        _, u, additional_info = DATA_add_noise(u_clean, data_obj_params)

        # ========================== save data ==================
        data_obj= DATA_data_construct(data_obj_params, u, u_clean, theta_D, theta_G)
        data_obj.additional_info = additional_info
        os.makedirs(data_path_dir, exist_ok=True)
        np.save(data_path_file, arr = data_obj, allow_pickle=True)  # save structure
        #print("Saved file as:",data_path_file)

    if additional_params["plot_bool"]:
        data_obj = np.load(data_path_file, allow_pickle=True).item()
        #err =  np.abs(data_obj.u -  data_obj.u_clean)
        #err_p = np.mean(err/data_obj.u_clean) * 100
        plot_1d_data(data_obj.x, data_obj.t, data_obj.u, data_obj.u_clean)