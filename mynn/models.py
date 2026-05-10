from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, lambda_list=None, dropout_p=0.0):
        self.input_shape = (1, 28, 28)
        self.conv_channels = 8
        self.hidden_dim = 128
        self.num_classes = 10
        self.lambda_list = lambda_list
        self.dropout_p = dropout_p

        self.conv = conv2D(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.relu1 = ReLU()
        self.fc1 = Linear(self.conv_channels * 14 * 14, self.hidden_dim)
        self.relu2 = ReLU()
        self.dropout = Dropout(dropout_p)
        self.fc2 = Linear(self.hidden_dim, self.num_classes)
        self.layers = [self.conv, self.relu1, self.fc1, self.relu2, self.dropout, self.fc2]

        if lambda_list is not None:
            optimizable_layers = [layer for layer in self.layers if layer.optimizable]
            for layer, weight_decay_lambda in zip(optimizable_layers, lambda_list):
                layer.weight_decay = True
                layer.weight_decay_lambda = weight_decay_lambda
        self.flatten_shape = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        if X.ndim == 2:
            outputs = X.reshape(X.shape[0], *self.input_shape)
        else:
            outputs = X

        outputs = self.conv(outputs)
        outputs = self.relu1(outputs)
        self.flatten_shape = outputs.shape
        outputs = outputs.reshape(outputs.shape[0], -1)
        outputs = self.fc1(outputs)
        outputs = self.relu2(outputs)
        outputs = self.dropout(outputs)
        outputs = self.fc2(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = self.fc2.backward(loss_grad)
        grads = self.dropout.backward(grads)
        grads = self.relu2.backward(grads)
        grads = self.fc1.backward(grads)
        grads = grads.reshape(self.flatten_shape)
        grads = self.relu1.backward(grads)
        grads = self.conv.backward(grads)
        return grads

    def train(self):
        self.dropout.train()

    def eval(self):
        self.dropout.eval()
    
    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        self.__init__(lambda_list=None, dropout_p=param_list.get('dropout_p', 0.0))
        for layer, params in zip([self.conv, self.fc1, self.fc2], param_list['params']):
            layer.W = params['W']
            layer.b = params['b']
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = params['weight_decay']
            layer.weight_decay_lambda = params['lambda']
        
    def save_model(self, save_path):
        param_list = {
            'model': 'Model_CNN',
            'input_shape': self.input_shape,
            'conv_channels': self.conv_channels,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'dropout_p': self.dropout_p,
            'params': [],
        }
        for layer in [self.conv, self.fc1, self.fc2]:
            param_list['params'].append({
                'W' : layer.params['W'],
                'b' : layer.params['b'],
                'weight_decay' : layer.weight_decay,
                'lambda' : layer.weight_decay_lambda,
            })

        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
