
# Medium-level
denoiseVFs = [0.4]
denoiseESs = [1000]
denoiseEpochs_lst = [1e6]*3

# Low-level
dataDiffLabels = ["const"]#, "quadratic", "exp"]
dataGrowLabels = ["zero"]
dataNoisePercents = [10]
denoiseUsizes = [64]
denoiseModelLabels = [0]#,1,2]
denoiseTVsplitSeeds = [0,1,2]
dataGammas = [1]
dataNoiseSeeds = [0]
combineDenoise = [0,1,2]

denoiseGenerateIndicesFuncLabel = "random"

# High level
# ============================================================
# loss
DNdataLossFuncLabel = "GLS"
denoiseManualTermination = None


# ==========================================
denoiseEpochs_lst = denoiseEpochs_lst[:len(denoiseTVsplitSeeds)]

def extend_a_with_b(a, b):
    a = np.array(a)
    b = np.array(b)
    
    missing = b[~np.isin(b, a)]  # elements in b but not in a
    if missing.size > 0:
        a = np.concatenate([a, missing])
    return a

# If using combine need to ensure there will be enough denoiseTVsplitSeeds
if combineDenoise:
    denoiseTVsplitSeeds = extend_a_with_b(denoiseTVsplitSeeds, combineDenoise)

