import numpy as np
import tqdm
import util
import copy

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU","output"]:   #output can be used for the final layer. Feel free to use/remove it
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        TODO
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, x):
        """
        args:
            x: np.array of shape (N, J) usually X times Weight Matrix
        return:
            np.array shaped (N, J) applied element wise sigmoid.
        """
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        args:
            x: np.array of shape (N, J) usually X times Weight Matrix
        return:
            np.array shaped (N, J) applied element wise tanh.
        """
        return np.tanh(x)

    def ReLU(self, x):
        """
        args:
            x: np.array of shape (N, J) usually X times Weight Matrix
        return:
            np.array shaped (N, J) applied element wise ReLU
        """
        return np.einsum("ij,ij->ij",x,x > 0) # Hadamard Product of x and boolean araryy

    def output(self, x):
        """
        args:
            x: np.array of shape (N, J) usually X times Weight Matrix
        return:
            np.array shaped (N, J) applied element wise Sofmax
        """
        exponent = np.exp(x)
        return np.divide(exponent.T,np.sum(exponent, axis=1).T).T

    def grad_sigmoid(self,x):
        """
        args:
            x: np.array of shape (N, J) usually X times Weight Matrix
        return:
            ...
        """
        return np.einsum("ij,ij->ij",self.sigmoid(x),1-self.sigmoid(x))

    def grad_tanh(self,x):
        """
        TODO: Compute the gradient for tanh here.
        """
        return 1 - np.power(self.tanh(x),2)

    def grad_ReLU(self,x):
        """
        TODO: Compute the gradient for ReLU here.
        """
        return (x > 0) + 0

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """

        return 1


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation, weightType):
        """
        args:
            in_unit: linear mapping W domain dimension
            out_unit: linear mapping W image dimension
            activation: callable Activation Object
            weightType: 'random' or np.array of shape = (in_unit,out_unit)
        """
        np.random.seed(42)
        if (weightType == 'random'):
            self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.x = None    # Save the input to forward in this
        self.a = None    #output without activation
        self.z = None    # Output After Activation
        self.activation = activation   #Activation function
        self.prev_w = np.zeros((in_units + 1, out_units))


        self.dw = 0  # Save the gradient w.r.t w in this. You can have bias in w itself or uncomment the next line and handle it separately
        # self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        TODO
        Make layer callable.
        No need to do?
        """
        self.x = x # may want to remove this
        return self.forward(util.append_bias(self.x))

    def forward(self, x):
        """
        args:
            x: Output of previous layer, np.array of shape = (N,J_(L+1) + 1_bias)
            J_(L-1) output dimension of L+1 th layer
            w: Weight matrix of shape (J_(L+1) + 1, J_L) linear mapping form J_(L+1) to J_L =
            a: matrix product of x w
        return:
            forward takes in an input vector x, computes the weighted input a and then uses its Activation class object to calculate the activation z.
        """
        self.a = np.einsum("nj,jk->nk",x,self.w)
        self.z = self.activation(self.a)
        return self.z

    def backward(self, deltaCur, learning_rate, momentum_gamma, regularization, gradReqd=True):
        """
        The function backward takes the weighted sum of the deltas (deltaCur variable in backward method of Activation class) from the layer above it as input,
        computes the gradient for its weights (to be saved in dw). If there is another layer below that (multiple hidden layers),
        it also passes the weighted sum of the deltas back to the previous layer. Otherwise, if the previous layer is the input layer, it stops there.

        args:
            deltaCur: weighted sum of the deltas np. deltaCur = B_(L-1) W_(L-1) passed from Network
            gradReqd: used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw

        Do:
            if Not Input Layer, update gradient w.r.t w_L and pass delta term to L-1 layer.
            if Input Layer, no need to pass delta.
        Delta Computataion:
            [deltaCur_(NxJ) x Weight_(JxK)] o (hadamard) grad_Activation
        """
        batch_size = len(self.x)
        if type(self.activation.backward(self.a)) == type(np.array([])):
            deltaNext = np.einsum("NK,NK->NK",deltaCur,self.activation.backward(self.a))
        else:
            deltaNext = deltaCur
        # self.dw =  np.einsum("NJ,NK->JK",util.append_bias(self.x),deltaNext)
        self.dw =  np.einsum("NJ,NK->JK",util.append_bias(self.x),deltaNext)/batch_size
        if gradReqd:
            temp_w = self.w
            self.w = self.w + (learning_rate) * self.dw + momentum_gamma * (self.w - self.prev_w) + regularization * 2 * self.w
            # self.w = self.w + (learning_rate/batch_size) * self.dw
            # self.w = self.w + learning_rate * self.dw
            self.prev_w = temp_w
        return deltaNext
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw
        The delta expression (that you prove in PA1 part1) for any layer consists of delta and weights from the next layer and derivative of the activation function
        of weighted inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative of the activation function of the weighted
        input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it and you are free to interpret it any other way.
        Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass
        """


class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.

    """

    def __init__(self, config):
        """
        TODO
        Create the Neural Network using config. Feel free to add variables here as per need basis
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None        # For saving the output vector of the model
        self.targets = None  # For saving the targets
        self.config = config

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation']),
                          config["weight_type"]))
            elif i == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output"),
                                         config["weight_type"]))

    def __call__(self, x, targets=None):
        """
        TODO
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)


    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        self.x = x
        self.targets = targets
        self.F = [x] # Empty list that stores all the forward pass
        for i in range(self.num_layers):
            self.F.append(self.layers[i](self.F[i])) # calling ith layer & append forward
        self.y = self.F[-1]
        return (self.loss(self.y,self.targets), util.calculateCorrect(self.y,self.targets))


    def loss(self, logits, targets):
        return -1*np.einsum("ij,ij",np.log(logits),targets)/len(targets)
        """
        Compute multiclass cross entropy.

        include momentum term and regularization term

        L(x) = - Σ (t*ln(y))

        Parameters
        ----------
        y
            The network's predictions
        t
            The corresponding targets
        Returns
        -------
        float 
            multiclass cross entropy loss value according to above definition
        """
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''

    def backward(self, gradReqd=True):
        deltaCur = self.targets - self.y #D_L/D_W_L
        for i in range(self.num_layers-1,-1,-1): # i = L-1,L-2,...,0
            w = copy.deepcopy(self.layers[i].w[1:,:])
            if i == 0:
                self.layers[i].backward(deltaCur,
                                        self.config['learning_rate'],
                                        self.config['momentum_gamma'],
                                        self.config['L2_penalty'],
                                        gradReqd)
            else:
                deltaCur = np.einsum("NK,JK->NJ",self.layers[i].backward(deltaCur,
                                                                         self.config['learning_rate'],
                                                                        self.config['momentum_gamma'],
                                                                        self.config['L2_penalty'],
                                                                        gradReqd),w) # Multiplying deltaNext & w
        '''
        TODO: Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''
