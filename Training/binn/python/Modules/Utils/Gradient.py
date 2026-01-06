# Version = 19 Dec 24,.
#
# Taken from [1].
#
# References
# ----------
# [1] Lagergren JH, Nardini JT, Baker RE, Simpson MJ, Flores KB (2020) Biologically-
#      informed neural networks guide mechanistic modeling from sparse experimental 
#      data. PLoS Comput Biol 16(12): e1008462. # https://doi.org/10.1371/journal.pcbi.1008462

from torch.autograd import grad

def Gradient(outputs, inputs, order=1):

    """
    Takes the gradient of outputs with respect to inputs up to some order.
    
    Inputs:
        outputs (tensor): function to be differentiated
        inputs  (tensor): differentiation argument
        order      (int): order of the derivative 
        
    Returns:
        grads   (tensor): 
    """
    
    # return outputs if derivative order is 0
    grads = outputs
    
    # convert outputs to scalar
    outputs = outputs.sum()

    # compute gradients sequentially until order is reached
    for i in range(order):
        grads = grad(outputs, inputs, create_graph=True)[0]
        outputs = grads.sum()

    return grads
