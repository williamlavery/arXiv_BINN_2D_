
# Low level
# ============================================================
binnUsizes = [64]
binnDsizes= [4]
binnGsizes = [4]
binnModelLabels = [0]
binnESs = [500]#, 1000, 2000]#, 3000, 5000]
gammas = [0]
noisePercents = [5]#, 5]
noiseSeeds = [0]
DoneParamBool = False  # True
#dataTnums = [5]

# split seeds
binnTVsplitSeeds = [0, 1, 2]
denoiseTVsplitSeeds = binnTVsplitSeeds
allConstraints = [True]


# Medium level
# ============================================================
binnVFs = [0.2]
binnEpochs = int(1e6)

cosAmplitude = 0.5
ICLabels = [f'cosFlat{cosAmplitude}']
diffLabels = ["const"]#, "linear","quadratic", "exp"]
growLabels = ["linear"]

twoStepBools = [0]#1
binnInitializeDenoiseBool = 0


# High level
# ============================================================
# pde loss
BCbool= 0
numPDEsamples = [100]
combineDenoise = None#[0,1,2]
binnES_check = [6000]
printFreq = 100

# data loss
BNdataLossFuncLabels = ["MSE"]


# =================== Denoise
# Medium-level
denoiseVF = 0.2
denoiseES = 500
denoiseEpochs_lst = [10000]

# High level
# ============================================================
# loss
denoiseManualTermination = None
DNdataLossFuncLabel = "MSE"



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




