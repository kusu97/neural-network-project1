from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        if initialize_method is np.random.normal:
            self.W = initialize_method(loc=0.0, scale=np.sqrt(2.0 / in_dim), size=(in_dim, out_dim))
        else:
            self.W = initialize_method(size=(in_dim, out_dim))
        self.b = np.zeros((1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        assert self.input is not None, "Forward must be called before backward."
        batch_size = self.input.shape[0]
        self.grads['W'] = self.input.T @ grad / batch_size
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True) / batch_size
        return grad @ self.W.T
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self.stride = stride
        self.padding = padding
        fan_in = in_channels * self.kernel_size * self.kernel_size
        if initialize_method is np.random.normal:
            self.W = initialize_method(
                loc=0.0,
                scale=np.sqrt(2.0 / fan_in),
                size=(out_channels, in_channels, self.kernel_size, self.kernel_size),
            )
        else:
            self.W = initialize_method(size=(out_channels, in_channels, self.kernel_size, self.kernel_size))
        self.b = np.zeros((1, out_channels, 1, 1))
        self.params = {'W' : self.W, 'b' : self.b}
        self.grads = {'W' : None, 'b' : None}
        self.input = None
        self.input_padded = None
        self.cols = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def _im2col(self, X):
        batch_size, channels, height, width = X.shape
        kernel = self.kernel_size
        out_h = (height - kernel) // self.stride + 1
        out_w = (width - kernel) // self.stride + 1
        cols = np.empty((batch_size, out_h, out_w, channels, kernel, kernel), dtype=X.dtype)
        for i in range(out_h):
            h_start = i * self.stride
            for j in range(out_w):
                w_start = j * self.stride
                cols[:, i, j, :, :, :] = X[:, :, h_start:h_start + kernel, w_start:w_start + kernel]
        return cols.reshape(batch_size * out_h * out_w, -1), out_h, out_w
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out, in, k, k]
        """
        self.input = X
        if self.padding > 0:
            self.input_padded = np.pad(
                X,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
            )
        else:
            self.input_padded = X

        self.cols, out_h, out_w = self._im2col(self.input_padded)
        W_col = self.W.reshape(self.out_channels, -1)
        output = self.cols @ W_col.T + self.b.reshape(1, self.out_channels)
        return output.reshape(X.shape[0], out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, _, out_h, out_w = grads.shape
        grad_reshaped = grads.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        self.grads['W'] = (grad_reshaped.T @ self.cols).reshape(self.W.shape) / batch_size
        self.grads['b'] = np.sum(grads, axis=(0, 2, 3), keepdims=True).reshape(1, self.out_channels, 1, 1) / batch_size

        W_col = self.W.reshape(self.out_channels, -1)
        dcols = grad_reshaped @ W_col
        dcols = dcols.reshape(
            batch_size,
            out_h,
            out_w,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        dX_padded = np.zeros_like(self.input_padded)
        for i in range(out_h):
            h_start = i * self.stride
            for j in range(out_w):
                w_start = j * self.stride
                dX_padded[:, :, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size] += dcols[:, i, j]

        if self.padding > 0:
            return dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dX_padded
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None
        self.probs = None
        self.grads = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        labels = labels.astype(np.int64)
        self.predicts = predicts
        self.labels = labels

        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        self.probs = np.clip(probs, 1e-12, 1.0)

        batch_size = predicts.shape[0]
        return -np.mean(np.log(self.probs[np.arange(batch_size), labels]))
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        self.grads = self.probs.copy()
        self.grads[np.arange(batch_size), self.labels] -= 1
        # Then send the grads to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition
