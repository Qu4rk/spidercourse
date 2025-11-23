import sys
sys.path.append('.')  # leave if you rely on running scripts directly
import numpy as np
from .activations import *

class Network:
    def __init__(self, seed: int, layers: list[int], activations: list) -> None:
        self.rng = np.random.default_rng(seed)
        self.layers = layers
        self.activations = activations

        self.weights = []
        self.biases = []
        self.preActivated = []
        self.activated = []
        self.gradw = []
        self.gradb = []
        self.deltas = []

        self.activationDeriv()
        self.initWeights()
        
    def __str__(self) -> str:
        # return a string with all weights and biases
        outLines = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            outLines.append(f"Layer {i} -> {i+1} weights shape {w.shape}:")
            outLines.append(str(w))
            outLines.append(f"Layer {i} -> {i+1} biases shape {b.shape}:")
            outLines.append(str(b))
        return "\n".join(outLines)
    
    def initWeights(self) -> None:
        for x in range(len(self.layers) - 1):
            n_in = self.layers[x]
            n_out = self.layers[x+1]
            limit = np.sqrt(6.0 / (n_in + n_out))
            w = self.rng.uniform(-limit, limit, (n_in, n_out))
            b = np.zeros((n_out,))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, input) -> np.ndarray[np.float64]:
        self.preActivated = []
        self.activated = []

        self.preActivated.append(input)
        self.activated.append(input)

        for x in range(len(self.layers) - 1):
            self.preActivated.append(self.activated[x].dot(self.weights[x]) + self.biases[x])
            self.activated.append(self.activations[x](self.preActivated[x+1]))

        #return last activated
        return self.activated[-1]

    def activationDeriv(self):
        self.activationDerivs = []
        for x in range(len(self.activations)):
            if self.activations[x] == ReLU:
                deriv = ReLUDeriv
            elif self.activations[x] == sigmoid:
                deriv = sigmoidDeriv
            elif self.activations[x] == step:
                deriv = stepDeriv
            elif self.activations[x] == softmax:
                deriv = softmaxDeriv
            elif self.activations[x] == Linear:
                deriv = LinearDeriv
            else:
                raise ValueError("Unknown activation function for derivative")
            self.activationDerivs.append(deriv)

    def predict(self, inputs) -> np.ndarray[np.float64]:
        return self.forward(inputs)

    def meanSquaredError(self, inputs, labels):
        diff = inputs-labels
        return 0.5*np.sum(np.power(diff, 2))
    
    def mseDeriv(self, inputs, labels):
        return inputs - labels

    def backward(self, loss):
        self.gradw = []
        self.gradb = []
        self.deltas = []

        self.deltas.append(loss * self.activationDerivs[-1](self.preActivated[-1]))
        self.gradw.append(self.activated[-2].T.dot(self.deltas[-1]))

        self.gradb.append(np.sum(self.deltas[-1], axis=0))

        for x in range(len(self.layers) - 2, 0, -1):
            dC_dA = self.deltas[-1].dot(self.weights[x].T)
            
            Z_hidden = self.preActivated[x]

            hidden_deriv_func = self.activationDerivs[x-1]

            delta = dC_dA * hidden_deriv_func(Z_hidden)
         
            self.deltas.append(delta)

            self.gradw.append(self.activated[x-1].T.dot(self.deltas[-1]))
            self.gradb.append(np.sum(self.deltas[-1], axis=0))
        
        self.gradw.reverse()
        self.gradb.reverse()
        self.deltas.reverse()

    def updateWeights(self, lr):
        for x in range(len(self.layers) - 1):
            self.weights[x] -= lr * self.gradw[x]
            self.biases[x] -= lr * self.gradb[x]
    
    def train(self, data: np.ndarray[np.float64], lr: float, epochs: int, testSplit: float = 0.2) -> list:
        """
        Train accepts data in either orientation:
        - rows-as-samples: shape (m, input_dim+output_dim)
        - cols-as-samples: shape (input_dim+output_dim, m)
        """
        print("Beginning Training....")

        if data.ndim != 2:
            raise ValueError("data must be a 2D numpy array")

        input_dim = self.layers[0]
        output_dim = self.layers[-1]
        total_cols = input_dim + output_dim

        # normalize orientation to rows-as-samples
        if data.shape[1] == total_cols:
            arr = data.copy()
        elif data.shape[0] == total_cols:
            arr = data.T.copy()
        else:
            raise ValueError(f"Data shape {data.shape} doesn't match network I/O dims; "
                             f"expected (m, {total_cols}) or ({total_cols}, m)")

        print("train: arr.shape =", arr.shape)
        m = arr.shape[0]
        # shuffle
        indices = np.arange(m)
        self.rng.shuffle(indices)
        arr = arr[indices]

        testLength = int(m * testSplit)
        testArr = arr[0:testLength]
        trainArr = arr[testLength:m]

        if trainArr.shape[0] == 0:
            raise ValueError("No training samples after split; reduce testSplit or provide more data")

        # Attempt to find the time/input column automatically if orientation got mixed
        # The time column was generated as linspace(0,1), so it should be within [0,1]
        col_mins = trainArr.min(axis=0)
        col_maxs = trainArr.max(axis=0)
        time_cols = np.where((col_mins >= -1e-6) & (col_maxs <= 1.000001))[0]

        if len(time_cols) == 1:
            time_idx = time_cols[0]
            print(f"Detected time column at index {time_idx}")
            trainFeatures = trainArr[:, [time_idx]]
            # all other columns are labels
            label_indices = [i for i in range(trainArr.shape[1]) if i != time_idx]
            trainLabels = trainArr[:, label_indices]
            if testArr.size:
                testFeatures = testArr[:, [time_idx]]
                testLabels = testArr[:, label_indices]
            else:
                testFeatures = np.empty((0, input_dim))
                testLabels = np.empty((0, output_dim))
        else:
            # fallback to assumed layout: first input_dim columns are features, rest labels
            trainFeatures = trainArr[:, :input_dim]
            trainLabels = trainArr[:, input_dim:]
            if testArr.size:
                testFeatures = testArr[:, :input_dim]
                testLabels = testArr[:, input_dim:]
            else:
                testFeatures = np.empty((0, input_dim))
                testLabels = np.empty((0, output_dim))

        print("trainFeatures.shape =", trainFeatures.shape, "trainLabels.shape =", trainLabels.shape)
        print("testFeatures.shape  =", testFeatures.shape, "testLabels.shape  =", testLabels.shape)

        losses = []

        for epoch in range(epochs):
            _ = self.forward(trainFeatures)  # sets self.activated[-1]
            loss_deriv = self.mseDeriv(self.activated[-1], trainLabels)
            self.backward(loss_deriv)
            self.updateWeights(lr)

            mse = self.meanSquaredError(self.activated[-1], trainLabels)
            avg_loss = mse / max(1, trainFeatures.shape[0])
            losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Avg Loss: {avg_loss:.6f}")

        # Evaluate on test set if present
        if testFeatures.size:
            _ = self.forward(testFeatures)
            norm_preds = self.activated[-1]
            norm_labels = testLabels

            total_loss = self.meanSquaredError(norm_preds, norm_labels)
            avg_test_loss = total_loss / max(1, testFeatures.shape[0])
            mae = np.mean(np.abs(norm_preds - norm_labels))
            ss_res = np.sum((norm_labels - norm_preds) ** 2)
            ss_tot = np.sum((norm_labels - np.mean(norm_labels)) ** 2)
            r_squared = 1.0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

            print(f"\n--- Test Results ---")
            print(f"Avg Test Loss (MSE): {avg_test_loss:.6f}")
            print(f"Mean Absolute Error (MAE): {mae:.6f}")
            print(f"R-squared (R²): {r_squared:.6f}")
        else:
            print("\nNo test samples provided (testSplit resulted in 0 test samples).")

        return losses