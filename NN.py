import numpy as np
from activations import *


#translated manually from https://github.com/AlanMet/DartNet
class Network:
    #seed: int
    #layers : [5, 5, 3]
    #[1, 64, 64, 24]
    #activations : [relu, relu]
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

        #dC/dz
        self.deltas.append(loss * self.activationDerivs[-1](self.preActivated[-1]))
        self.gradw.append(self.activated[-2].T.dot(self.deltas[-1]))

        self.gradb.append(np.sum(self.deltas[-1], axis=0))

        for x in range(len(self.layers) - 2, 0, -1):
            # 1. This correctly calculates dC/dA for the current layer
            dC_dA = self.deltas[-1].dot(self.weights[x].T)
            
            # --- FIX 2 ---
            # 2. Get the Z value of the current hidden layer
            Z_hidden = self.preActivated[x]
            
            # 3. Get the deriv func (note: x-1, e.g. layer 0's activ)
            hidden_deriv_func = self.activationDerivs[x-1]
            
            # 4. Calculate the true delta (dC/dZ)
            delta = dC_dA * hidden_deriv_func(Z_hidden)
            
            # 5. Now, append the correct value to self.deltas
            self.deltas.append(delta)
            # --- END FIX 2 ---
            
            # This line is now correct
            self.gradw.append(self.activated[x-1].T.dot(self.deltas[-1]))
            self.gradb.append(np.sum(self.deltas[-1], axis=0))
        
        self.gradw.reverse()
        self.gradb.reverse()
        self.deltas.reverse()

    def updateWeights(self, lr):
        for x in range(len(self.layers) - 1):
            self.weights[x] -= lr * self.gradw[x]
            self.biases[x] -= lr * self.gradb[x]
    
    def train(self, data: np.ndarray[np.float64], lr: float, epochs: int, testSplit: float) -> None:
        print("Beginning Training....")
        #train test split
        m, n = data.shape
        np.random.shuffle(data)

        testLength = int(m * testSplit)

        testData = data[0:testLength].T
        testLabels = testData[0].reshape(-1, 1)
        testFeatures = testData[1:n].T

        trainData = data[testLength: m].T
        trainLabels = trainData[0].reshape(-1, 1)
        trainFeatures = trainData[1:n].T
        # print(trainFeatures.shape)

        for x in range(epochs):
            forward = self.forward(trainFeatures)
            loss = self.mseDeriv(self.activated[-1], trainLabels)
            self.backward(loss)
            self.updateWeights(lr)
            if x % 10 == 0:
                mse = self.meanSquaredError(self.activated[-1], trainLabels)
                avg_loss = mse / trainFeatures.shape[0]
                print(f"Epoch {x}, Avg Loss: {avg_loss}")

        #test
        # 1. Run the forward pass on the TEST features
        forward_test = self.forward(testFeatures)

        # 2. Get the normalized predictions and labels
        #    (self.activated[-1] was set by the line above)
        norm_preds = self.activated[-1]
        norm_labels = testLabels

        # 3. Calculate Avg Test Loss (Mean Squared Error)
        #    This is what you already had.
        total_loss = self.meanSquaredError(norm_preds, norm_labels)
        avg_test_loss = total_loss / testFeatures.shape[0]

        # --- START NEW ACCURACY METRICS ---
        
        # 4. Calculate Mean Absolute Error (MAE)
        #    This is "on average, how far off are we in normalized units?"
        mae = np.mean(np.abs(norm_preds - norm_labels))

        # 5. Calculate R-squared (R²)
        #    This is the "goodness of fit" score from 0 to 1.
        sum_of_squares_residual = np.sum((norm_labels - norm_preds) ** 2)
        sum_of_squares_total = np.sum((norm_labels - np.mean(norm_labels)) ** 2)
        
        # Handle the rare case of perfect prediction (to avoid division by zero)
        if sum_of_squares_total == 0:
            r_squared = 1.0
        else:
            r_squared = 1 - (sum_of_squares_residual / sum_of_squares_total)

        # --- END NEW ACCURACY METRICS ---


        # Print all meaningful metrics
        print(f"\n--- Test Results ---")
        print(f"Avg Test Loss (MSE): {avg_test_loss:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"R-squared (R²): {r_squared:.6f}")
        
        # Return the R² score as the "accuracy"
        return r_squared