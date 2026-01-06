# ========================================== generate data object for saving ==================================
def BN_save_model_generated_dataObj(model_loaded, dataobj, model_label, path_to_dataObj_dir, device):

    Nt, Nx1, Nx2 = len(dataobj.t), len(dataobj.x1), len(dataobj.x2)
    model_loaded.load_best_val()

    with torch.no_grad():
        u_pred_flat_torch = model_loaded.model(to_torch_grad(dataobj.inputs, device))  # Use direct net call (no ID input)
        u  = u_pred_flat_torch.reshape(Nx1, Nx2, Nt).detach().cpu().numpy()
    dataObj_name = f"data_binn_num{model_label}"
    save_path = os.path.join(path_to_dataObj_dir, dataObj_name)
    data_obj_binn = Data(
                            x1= dataobj.x1,
                            x2= dataobj.x2,
                            t =dataobj.t,
                            u_clean = [],
                            u = u,
                            gamma = dataobj.gamma,
                            theta_D = dataobj.theta_D,
                            theta_G = dataobj.theta_G)

    np.save(save_path, data_obj_binn) # automatically overwrites if exists
    print(f"Saved data_obj:'{dataObj_name}'")

