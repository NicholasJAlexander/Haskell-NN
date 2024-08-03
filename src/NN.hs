module NN where
import System.Random (randomRIO)
import Control.Monad (replicateM)
import Matrix
import System.IO.Unsafe (unsafePerformIO)
import ActivationFunctions (ActivationFunc, aF, aF')
import qualified Data.Vector.Unboxed as UV
import qualified Data.List as L

-- * Weight and Bias Generation

-- | Function to generate a matrix with random values between a and b, appearing as pure.
generateWeightsAndBiases :: (Int, Int) -> Double -> Double -> (Matrix Double, ColumnVector Double)
generateWeightsAndBiases (n, m) a b = unsafePerformIO $ do
    weights <- fromLists <$> replicateM n (replicateM m (randomRIO (a, b)))
    biases <- cvFromList <$> replicateM n (randomRIO (a, b))
    return (weights, biases)

-- | Generate weight matrices and bias vectors for each layer in the network.
generateWeightsMatricesAndBiases :: [Int] -> Double -> Double -> [(Matrix Double, ColumnVector Double)]
generateWeightsMatricesAndBiases neurons a b =
    zipWith (\m n -> generateWeightsAndBiases (m, n) a b) (tail neurons) neurons

-- * Data Types

-- | Data type representing a layer in a neural network.
--   Contains the weights matrix, biases, and the activation function used by the layer.
data Layer = Layer {
    lWeights :: Matrix Double,
    lBiases :: ColumnVector Double,
    lAF :: ActivationFunc
}

instance Show Layer where
    show (Layer weights biases af) = 
        "Layer { weights = " ++ show (nrows weights) ++ "x" ++ show (ncols weights) ++
        " matrix, biases = " ++ show (cvLength biases) ++ 
        " vector, activation = " ++ show af ++ " }"

-- | Data type representing a backpropagation neural network.
--   Contains a list of network layers and a learning rate.
data BackpropNet = BackpropNet {
    -- | List of layers in the neural network.
    layers :: [Layer],
    -- | The learning rate of the neural network.
    learningRate :: Double
}

instance Show BackpropNet where
    show net =
        "BackpropNet { " ++
        "layers = " ++ show (length (layers net)) ++ " layers, " ++
        "neurons = " ++ show (getNeuronCounts net) ++ ", " ++
        "learningRate = " ++ show (learningRate net) ++ " }"
      where
        getNeuronCounts network =
            (ncols . lWeights . head . layers $ network) : 
            map (nrows . lWeights) (layers network)

-- | Data type representing a layer during the propagation phase in a neural network.
--   Contains various properties required for forward and backward propagation.
data PropagatedLayer = PropagatedLayer {
    -- | Input to the layer. Represented as a column vector.
    propIn :: ColumnVector Double,
    -- | Output from the layer. Also a column vector.
    propOut :: ColumnVector Double,
    -- | First derivative of the activation function for this layer.
    propF'a :: ColumnVector Double,
    -- | Weights for this layer represented as a matrix.
    propW :: Matrix Double,
    -- | Biases for this layer.
    propB :: ColumnVector Double,
    -- | Activation function used by this layer.
    propAF :: ActivationFunc
}

-- | Data type representing a layer during the backpropagation phase in a neural network.
--   Contains all the necessary components for the backward pass of the training algorithm.
data BackpropagatedLayer = BackpropagatedLayer {
    -- | Partial derivative of the cost with respect to the z-value (weighted input) of this layer.
    bpDazzle :: ColumnVector Double,
    -- | Gradient of the biases for this layer.
    bpBiasGrad :: ColumnVector Double,
    -- | Gradient of the error with respect to the output of this layer.
    bpErrGrad :: Matrix Double,
    -- | Value of the first derivative of the activation function for this layer.
    bpF'a :: ColumnVector Double,
    -- | Input to this layer.
    bpIn :: ColumnVector Double,
    -- | Output from this layer.
    bpOut :: ColumnVector Double,
    -- | Weights for this layer.
    bpW :: Matrix Double,
    -- | Activation function specification for this layer.
    bpAF :: ActivationFunc
}

-- * Helper Functions

-- | Checks if the dimensions of two matrices are compatible for multiplication.
--   Throws an error if dimensions are inconsistent.
dimensionMatch :: Matrix Double -- ^ Matrix 1 (to be checked for matching dimensions with Matrix 2).
               -> Matrix Double -- ^ Matrix 2 (returned if dimensions with Matrix 1 match).
               -> Matrix Double -- ^ The second matrix, if the dimensions match.
dimensionMatch m1 m2
    | nrows m1 == ncols m2 = m2
    | otherwise = error "Inconsistent dimensions in weight matrix"

-- | Validates the input vector for a neural network.
validateInput :: BackpropNet         -- ^ The neural network for input validation.
              -> ColumnVector Double -- ^ The input vector to be validated.
              -> ColumnVector Double -- ^ The validated input vector; throws error if validation fails.
validateInput net input
    | validSize = input
    | otherwise = error $ "Inconsistent dimensions in input vector. Expected size: " ++
        show fLayerSize ++
        ", Actual size: " ++
        show inputSize
    where
        validSize = fLayerSize == inputSize
        inputSize = cvLength input
        fLayerSize = ncols (lWeights (head (layers net)))

-- | Validates the target vector for a neural network layer.
validateTarget :: PropagatedLayer
               -> ColumnVector Double
               -> ColumnVector Double
validateTarget pLayer target
    | targetLength == expectedLength = target
    | otherwise = error $ "Inconsistent dimensions in target vector. Expected size: " ++
        show expectedLength ++
        ", Actual size: " ++
        show targetLength
    where
        targetLength = cvLength target
        expectedLength = cvLength (propOut pLayer)

-- * Propagation

-- | Propagates the state of one layer to the next in a neural network.
propagate :: PropagatedLayer -> Layer -> PropagatedLayer
propagate layerJ layerK = PropagatedLayer {
    propIn = x,
    propOut = y,
    propF'a = f'a,
    propW = w,
    propB = b,
    propAF = lAF layerK
    }
    where
        x = propOut layerJ
        w = lWeights layerK
        b = lBiases layerK
        a = cvAdd (matrixVectorMult w x) b
        f = aF (lAF layerK)
        y = f a
        f' = aF' (lAF layerK)
        f'a = f' a

-- | Propagates the input layer of a neural network.
propagateInputLayer :: ColumnVector Double -> Layer -> PropagatedLayer
propagateInputLayer input layerK = PropagatedLayer {
    propIn = x,
    propOut = y,
    propF'a = f'a,
    propW = w,
    propB = b,
    propAF = lAF layerK
    }
    where
        x = input
        w = lWeights layerK
        b = lBiases layerK
        a = cvAdd (matrixVectorMult w x) b
        f = aF (lAF layerK)
        y = f a
        f' = aF' (lAF layerK)
        f'a = f' a

-- | Propagates an input vector through the neural network, returning the state of each layer after propagation.
propagateNet :: ColumnVector Double -- ^ The input vector to the neural network.
             -> BackpropNet         -- ^ The backpropagation neural network through which the input is propagated.
             -> [PropagatedLayer]   -- ^ A list of 'PropagatedLayer' representing the state of each layer after the input is propagated through them.
propagateNet input backNet = scanl propagate pFirstlayer remainingLayers
    where
        netLayers = layers backNet
        remainingLayers = tail netLayers
        pFirstlayer = propagateInputLayer validInput (head netLayers)
        validInput = validateInput backNet input

-- * Backpropagation

-- | Performs backpropagation for a single layer of a neural network.
backpropagate :: PropagatedLayer      -- ^ The forward propagated state of the current layer.
              -> BackpropagatedLayer  -- ^ The backpropagated state of the next layer.
              -> BackpropagatedLayer  -- ^ The backpropagated state of the current layer after computing necessary values.
backpropagate layerJ layerK = BackpropagatedLayer {
    bpDazzle = dazzleJ,
    bpBiasGrad = dazzleJ,
    bpErrGrad = errorGrad dazzleJ f'aJ (propIn layerJ),
    bpF'a = propF'a layerJ,
    bpIn = propIn layerJ,
    bpOut = propOut layerJ,
    bpW = propW layerJ,
    bpAF = propAF layerJ
    }
    where
        dazzleJ = customMatrixVectorMult wKT dazzleK f'aK
        dazzleK = bpDazzle layerK
        wKT = transpose (bpW layerK)
        f'aK = bpF'a layerK
        f'aJ = propF'a layerJ

-- | Computes the gradient of the error with respect to the layer's weights.
errorGrad :: ColumnVector Double -> ColumnVector Double -> ColumnVector Double -> Matrix Double
errorGrad dazzle f'a input =
    generateMatrix (cvLength dazzle) $ \i ->
        cvGenerate (cvLength input) $ \j ->
            (cvGetElem dazzle i * cvGetElem f'a i) * cvGetElem input j

-- | Performs the backpropagation step for the final (output) layer of a neural network.
backpropagateFinalLayer :: PropagatedLayer      -- ^ The forward propagated state of the output layer.
                        -> ColumnVector Double  -- ^ The target output vector.
                        -> BackpropagatedLayer  -- ^ The backpropagated state of the output layer.
backpropagateFinalLayer layerK target = BackpropagatedLayer {
    bpDazzle = dazzle,
    bpBiasGrad = dazzle,
    bpErrGrad = errorGrad dazzle f'a (propIn layerK),
    bpF'a = propF'a layerK,
    bpIn = propIn layerK,
    bpOut = propOut layerK,
    bpW = propW layerK,
    bpAF = propAF layerK
    }
    where
        dazzle = UV.zipWith (-) (propOut layerK) target
        validTarget = validateTarget layerK target
        f'a = propF'a layerK

-- | Executes backpropagation across all layers of a neural network.
backpropagateNet :: ColumnVector Double            -- ^ The target output vector for the entire network.
                 -> [PropagatedLayer]              -- ^ List of layers in their forward propagated state.
                 -> [BackpropagatedLayer]          -- ^ List of layers in their backpropagated state.
backpropagateNet target propLayers = L.scanr backpropagate bpOutputLayer hiddenLayers
    where
        hiddenLayers = init propLayers
        bpOutputLayer = backpropagateFinalLayer (last propLayers) target

-- * Weight Update

-- | Updates the weights of a layer in a neural network based on the backpropagation results.
update :: Double              -- ^ The learning rate, determining how much the weights are adjusted.
       -> BackpropagatedLayer -- ^ The backpropagated layer containing the error gradient and current weights.
       -> Layer               -- ^ The updated layer with new weights.
update rate layer = Layer { lWeights = wNew, lBiases = bNew, lAF = bpAF layer }
    where
        wNew = elementwise (\w g -> w - rate * g) (bpW layer) (bpErrGrad layer)
        bNew = cvZipWith (\b g -> b - rate * g) (bpBiasGrad layer) (bpBiasGrad layer)

-- | Updates the weights of all layers in the network.
updateLayers :: Double               -- ^ The learning rate.
             -> [BackpropagatedLayer] -- ^ List of backpropagated layers.
             -> [Layer]               -- ^ List of updated layers.
updateLayers _ [] = []
updateLayers rate (l:ls) = update rate l : updateLayers rate ls

-- * Network Initialization

-- | Checks dimensions of weight matrices and biases for consistency.
checkDimensions :: (Matrix Double, ColumnVector Double) -> (Matrix Double, ColumnVector Double) -> (Matrix Double, ColumnVector Double)
checkDimensions (w1, _) (w2, b2)
    | nrows w1 == ncols w2 && nrows w2 == cvLength b2 = (w2, b2)
    | nrows w1 /= ncols w2 = error "Inconsistent dimensions between weight matrices"
    | nrows w2 /= cvLength b2 = error "Inconsistent dimensions between weight matrix and bias vector"
    | otherwise = error "Unknown dimension inconsistency"

-- | Builds a backpropagation neural network with the given learning rate, weights, biases, and activation functions.
buildBackpropNet :: Double -> [(Matrix Double, ColumnVector Double)] -> [ActivationFunc] -> BackpropNet
buildBackpropNet lr wsAndBs aFs = BackpropNet { layers = networkLayers, learningRate = lr }
    where
        checkedWeightsAndBiases = scanl1 checkDimensions wsAndBs
        networkLayers = zipWith createLayer wsAndBs aFs
        createLayer (weights, biases) af = Layer { lWeights = weights, lBiases = biases, lAF = af }

-- | Function to initialize a neural network.
initializeNetwork :: [Int] -> [ActivationFunc] -> Double -> Double -> Double -> BackpropNet
initializeNetwork neurons activationFuncs a b lRate =
    buildBackpropNet lRate weightsAndBiases activationFuncs
    where
        weightsAndBiases = generateWeightsMatricesAndBiases neurons a b
        networkLayers = zipWith createLayer weightsAndBiases activationFuncs
        createLayer (weights, biases) af = Layer { lWeights = weights, lBiases = biases, lAF = af }

-- * Training

-- | Trains the network on a single example.
trainSingleExample :: BackpropNet -> (ColumnVector Double, ColumnVector Double) -> BackpropNet
trainSingleExample currentNet (input, target) = currentNet { layers = updatedLayers }
    where
        propagatedLayers = propagateNet input currentNet
        backpropagatedLayers = backpropagateNet target propagatedLayers
        updatedLayers = updateLayers (learningRate currentNet) backpropagatedLayers

-- | Trains the network on a batch of data.
trainBatch :: BackpropNet -> [(ColumnVector Double, ColumnVector Double)] -> BackpropNet
trainBatch = L.foldl' trainSingleExample

-- | Trains the network for a specified number of epochs.
trainEpochs :: BackpropNet -> Int -> [(ColumnVector Double, ColumnVector Double)] -> BackpropNet
trainEpochs net epochs dataset = iterate (`trainBatch` dataset) net !! epochs

-- * Output Functions

-- | Prints the weight matrices of the layers.
printLayers :: [Layer] -> [IO ()]
printLayers = map (printMatrix . lWeights)

-- | Prints the biases of the layers.
printLayerBiases :: [Layer] -> [IO ()]
printLayerBiases = map (print . lBiases)

-- | Gets the output of the neural network for a given input.
getOutput :: BackpropNet -> ColumnVector Double -> ColumnVector Double
getOutput net input = extractOutput (propagateNet input net)
    where
        extractOutput propLayers = propOut (last propLayers)

-- | Predicts the class for a given input.
predictClass :: BackpropNet -> ColumnVector Double -> Int
predictClass net input = UV.maxIndex (getOutput net input)
