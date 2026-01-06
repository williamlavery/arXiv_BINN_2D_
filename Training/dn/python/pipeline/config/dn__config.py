# ==========================================================================================
# Fixed
ext  = "Training/dataObj/original?_1/DValue_0.024720000000000002/rValue_1.536/gamma_0.2/K_1700.0/xNum_38/tNum_5/originalDataObj.npy"
denoiseDevice = 'cpu'
# ==========================================================================================
# ==========================================================================================

# High-level
dataICLabel = 'cos'


denoiseBatchSize = 38
denoiseRelUpdateThresh = 0.05
denoiseRelSaveThresh = 0.05
denoiseLR = 1e-3

plot_bool = 0
overwrite_bool = 0