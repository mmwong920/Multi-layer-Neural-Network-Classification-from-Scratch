import numpy as np
from neuralnet import Neuralnetwork
import copy

def check_grad(model, x_train, y_train):
    models = copy.deepcopy((model,)*6) # Tuple containing 6 deep copies of model
    layer_index = [-1,1,-2,-2,0,0]
    weight_index = [0,0,0,0,1,0,2,0,1,0,2,0]
    epsilon = 1e-2
    numerical_grad = []
    back_prop_grad = []
    for i in range(6):
        grad = model.layers[layer_index[i]].dw[2*i,2*i + 1] # Retrieving the gradient for w
        models[i].layers[layer_index[i]].w[2*i,2*i + 1] += epsilon
        loss_plus_e , accuracy = models[i].forward(x_train,y_train)
        models[i].layers[layer_index[i]].w[2*i,2*i + 1] -= 2*epsilon
        loss_minus_e , accuracy = models[i].forward(x_train,y_train)
        numerical_grad.append((loss_plus_e - loss_minus_e)/(2*epsilon))
        back_prop_grad.append(grad)
    return numerical_grad, back_prop_grad,  abs(np.array(back_prop_grad) - np.array(numerical_grad))
    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation

        return tuple of lists containing:
            gradient obtained from numerical approximation,
            gradient obtained by backpropagation,
            absolute difference between the two
    """



def checkGradient(x_train,y_train,config):

    subsetSize = 10  #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    model.forward(x_train_sample,y_train_sample)
    model.backward(gradReqd=False)
    print(check_grad(model, x_train_sample, y_train_sample)[-1])
    return check_grad(model, x_train_sample, y_train_sample)[-1]