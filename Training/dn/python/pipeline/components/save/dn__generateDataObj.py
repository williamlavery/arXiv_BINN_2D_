import re, copy
from pathlib import Path
# ========================================== generate data object for saving ==================================
def DN_save_model_generated_dataObj(model_loaded, dataobj, model_label, path_to_dataObj_dir, device):

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



# def average_dn_models(paths_to_combine,model_label = 0, termination_bool=True):


#     """
#     paths_to_combine: list of model file paths (.pth or checkpoint files)
#     model_constructor: a function or class to instantiate a fresh model

#     Returns:
#         averaged_model: a model with averaged parameters
#     """
#     avg_state_dict = None
#     num_models = 0
#     denoiseModelLabel = 0
#     u_error_bests = []

#     for data_path in paths_to_combine:
#         model_path = os.path.join(data_path, f"denoiseModel{denoiseModelLabel}.pth")
#         # Load model checkpoint
#         model = torch.load(model_path,weights_only=False)
#         model.load_best_terminated() if termination_bool else model.load_best_val()
#         state_dict = model.model.surface_fitter.state_dict()

#         # Accumulate weights
#         if avg_state_dict is None:
#             avg_state_dict = copy.deepcopy(state_dict)
#             for key in avg_state_dict:
#                 avg_state_dict[key] = avg_state_dict[key].clone()  # clone tensors
#         else:
#             for key in avg_state_dict:
#                 avg_state_dict[key] += state_dict[key]
#         num_models += 1
#         u_error_bests.append(model.u_error_best)
    
#     # Average the parameters
#     for key in avg_state_dict:
#         avg_state_dict[key] /= num_models

#     # Load into a fresh model
#     u_MLP_dn_avg = u_MLP_dn(u_size=model.model.surface_fitter.size)
#     u_MLP_dn_avg.load_state_dict(avg_state_dict)
#     u_pred = model.model.u_scale * u_MLP_dn_avg(model.model.torch_meshgrid).flatten()
#     u_avg_error = (u_pred - model.u_clean_torch_flat)**2
#     u_MLP_dn_avg.u_avg_error = torch.mean(u_avg_error).item()
#     print("Individual errors:", [f"{err:.3e}" for err in u_error_bests])
#     print(f"Best individual error {np.min(u_error_bests):.3e}")
#     print(f"Ensemble error {u_MLP_dn_avg.u_avg_error:.3e}")

#     return u_MLP_dn_avg, model.u_clean_torch_flat.detach().cpu().numpy()

# def DN_generate_ensemble_dataObj(model_label,
#                                  path_to_dataObj_dir,
#                                  data_obj_params, 
#                                  TV_params,
#                                   model_params, 
#                                   fit_params):


#     additional_params = data_obj_params["additional_params"]["denoise_path"]
#     denoise_path = data_obj_params["additional_params"]["denoise_path"]
#     combineDenoise = fit_params["denoise_fit_params_additionals"]["combineDenoise"]
#     TV_params["denoiseTV_params"]["denoiseGenerateIndicesArgs"]["denoiseTVsplitSeed"] = denoise_fit_params_additionals["combineDenoise"]

#     merged_path, _ = DN_model_finder(denoise_path, data_obj_params, TV_params, model_params, fit_params, mkdir=True)

#     paths_to_combine = [] 
#     avg_num = len(combineDenoise)
#     u_avg = 0

#     for denoiseTVsplitSeed in combineDenoise:
#         TV_params["denoiseTV_params"]["denoiseGenerateIndicesArgs"]["denoiseTVsplitSeed"] = denoiseTVsplitSeed 
#         individual_path, _ = DN_model_finder(denoise_path, data_obj_params, TV_params, model_params, fit_params,mkdir=False)
#         data_path = os.path.join(individual_path, f"data_dn_num{model_label}.npy")
#         try:
#             data_obj = np.load(data_path, allow_pickle=True).item(0)
#         except:
#             return "Can'combine yet as not all models needed for the combination have been produced"
#         u_avg += data_obj.u/avg_num
#         paths_to_combine.append(individual_path)
#         #plt.plot(data_obj.u[:,0], label=f"indiv{denoiseTVsplitSeed}", lw=1)

#     #plt.plot(u_avg[:,0], label=f"avg")


#     # generated ensemble model
#     termination_bool = TV_params["denoiseTV_params"]["denoiseVF"] and fit_params["denoise_fit_params"]["denoiseManualTermination"]
#     averaged_model, u_clean_np = average_dn_models(paths_to_combine, termination_bool=termination_bool)

#     # Save ensemble model
#     model_save_path_ensemble = os.path.join(merged_path, f"denoiseModelEnsemble{model_label}.pth")
#     torch.save(averaged_model,model_save_path_ensemble)
#     print(f"Ensemble dn model:'{ model_save_path_ensemble}'")


#     # Save data object
#     data_obj.u = u_avg
#     save_path_ensemble = os.path.join(merged_path, f"data_dnEnsemble_num{model_label}.npy")
#     np.save(save_path_ensemble, data_obj)
#     print(f"Ensemble data_obj:'{dataObj_name_ensemble}'")

#     #plt.plot(u_clean_np.reshape(-1,5)[:,0], label=f"clean", color='k', lw=1)
#     #plt.legend()
#     #plt.show()


