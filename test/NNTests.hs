{-# LANGUAGE FlexibleInstances #-}

module Main where

import Test.QuickCheck
import NN
import Matrix
import ActivationFunctions (ActivationFunc(..), tanhAF, sigmoidAF)

-- * Newtype Wrappers

-- | Wrapper for a 'Layer' to use with QuickCheck.
newtype TestLayer = TestLayer { getLayer :: Layer }

-- | Wrapper for an 'ActivationFunc' to use with QuickCheck.
newtype TestActivationFunc = TestActivationFunc { getActivationFunc :: ActivationFunc }

-- | Wrapper for a 'BackpropNet' to use with QuickCheck.
newtype TestBackpropNet = TestBackpropNet { getBackpropNet :: BackpropNet }

instance Show TestBackpropNet where
    show network =
        let net = getBackpropNet network
        in "BackpropNet { " ++
           "layers = " ++ show (length (layers net)) ++ " layers, " ++
           "neurons = " ++ show (getNeuronCounts net) ++ ", " ++
           "learningRate = " ++ show (learningRate net) ++ " }"
      where
        getNeuronCounts net =
            (ncols . lWeights . head . layers $ net) :
            map (nrows . lWeights) (layers net)

-- * Arbitrary Instances

-- | Arbitrary instance for 'ColumnVector Double', avoiding zero values.
instance Arbitrary (ColumnVector Double) where
  arbitrary = do
    size <- choose (1, 10)
    items <- cvReplicateM size (choose (0.1, 0.5))
    return $ cvFromList (cvToList items)

-- | Arbitrary instance for 'Matrix Double', avoiding zero values.
instance Arbitrary (Matrix Double) where
  arbitrary = do
    rows <- choose (1, 10)
    cols <- choose (1, 10)
    values <- vectorOf (rows * cols) (choose (0.1, 0.2))
    return $ fromLists [take cols $ drop (i * cols) values | i <- [0..rows-1]]

-- | Arbitrary instance for 'TestActivationFunc' using a newtype wrapper.
instance Arbitrary TestActivationFunc where
  arbitrary = TestActivationFunc <$> elements [sigmoidAF]

-- | Arbitrary instance for 'TestLayer' using a newtype wrapper.
instance Arbitrary TestLayer where
  arbitrary = do
    weights <- arbitrary
    biases <- arbitrary
    TestActivationFunc activationFunc <- arbitrary
    return $ TestLayer $ Layer weights biases activationFunc

-- | Arbitrary instance for 'TestBackpropNet' using a newtype wrapper.
instance Arbitrary TestBackpropNet where
  arbitrary = do
    numLayers <- choose (1, 5)  -- Networks with 1 to 5 layers
    layers <- vectorOf numLayers arbitrary
    learningRate <- choose (0.01, 0.05)
    return $ TestBackpropNet $ BackpropNet (map getLayer layers) learningRate

-- * Generators

-- | Generator for a list of neurons (layer sizes).
genNeurons :: Gen [Int]
genNeurons = do
  numLayers <- choose (3, 5)  -- Choose between 3 and 5 layers
  vectorOf numLayers (choose (1, 10))  -- Each layer has 1 to 10 neurons

-- | Generator for a list of activation functions.
genActivationFuncs :: Int -> Gen [ActivationFunc]
genActivationFuncs numLayers =
  vectorOf (numLayers - 1) (elements [tanhAF])

-- | Generator for a 'BackpropNet'.
genBackpropNet :: Gen TestBackpropNet
genBackpropNet = do
  neurons <- genNeurons
  activationFuncs <- genActivationFuncs (length neurons)
  a <- choose (-0.1, 0)
  b <- choose (0, 0.1)
  learningRate <- choose (0.05, 0.1)
  return $ TestBackpropNet $ initializeNetwork neurons activationFuncs a b learningRate

-- | Generator for input-target pairs.
genInputTargetPair :: BackpropNet -> Gen (ColumnVector Double, ColumnVector Double)
genInputTargetPair net = do
    let inputSize = ncols . lWeights . head $ layers net
    let outputSize = nrows . lWeights . last $ layers net
    input <- vectorOf inputSize (choose (-1, 1))
    targetIndex <- choose (0, outputSize - 1)
    let target = cvFromList $ replicate targetIndex 0 ++ [1] ++ replicate (outputSize - targetIndex - 1) 0
    return (cvFromList input, target)

-- * Property Tests

-- | Property to test that propagation preserves the output size.
prop_propagationPreservesSize :: Property
prop_propagationPreservesSize = forAll genBackpropNet $ \(TestBackpropNet net) ->
  not (null (layers net)) ==>
  forAll (vectorOf (inputSize net) (choose (0.1, 0.5))) $ \input ->
    collect (length $ layers net) $  -- Number of layers
    collect (inputSize net) $        -- Input size     
    collect (lastLayerSize net) $    -- Output size    
    let prevsize_output = getOutput net (cvFromList input)
    in cvLength prevsize_output == lastLayerSize net
  where
    inputSize net = ncols . lWeights . head $ layers net
    lastLayerSize net = nrows . lWeights . last $ layers net

-- | Mean squared error (MSE) calculation.
mse :: ColumnVector Double -> ColumnVector Double -> Double
mse output target =
    let diff = cvZipWith (-) output target
        squared = cvMap (\x -> x * x) diff
    in cvSum squared / fromIntegral (cvLength output)

-- | Property to test the sensitivity of the network to small input changes.
prop_inputSensitivity :: Property
prop_inputSensitivity = forAll genBackpropNet $ \(TestBackpropNet net) ->
  forAll (genInputTargetPair net) $ \(input, _) ->
    let output1 = getOutput net input
        perturbedInput = cvMap (+ 1e-5) input
        output2 = getOutput net perturbedInput
    in mse output1 output2 < 0.01

-- | Property to test that the network is deterministic.
prop_deterministic :: Property
prop_deterministic = forAll genBackpropNet $ \(TestBackpropNet net) ->
  forAll (genInputTargetPair net) $ \(input, _) ->
    getOutput net input == getOutput net input

-- | Property to test that training reduces error.
prop_trainingReducesError :: Property
prop_trainingReducesError = forAll genBackpropNet $ \(TestBackpropNet network) ->
  let modifiedNet = network {learningRate = 0.000000001} in
  forAll (genInputTargetPair modifiedNet) $ \(input, target) ->
    let unTrainedOutput = getOutput modifiedNet input
        trainedOutput = getOutput (trainBatch modifiedNet [(input, target)]) input
        errorBefore = mse target unTrainedOutput
        errorAfter = mse target trainedOutput
    in counterexample ("Output before training: " ++ show unTrainedOutput ++ show (errorBefore) ++
                       "\nOutput after training: " ++ show trainedOutput ++ show (errorAfter)++
                       "\nTarget: " ++ show target) $ errorAfter < errorBefore || errorAfter < 0.01

-- | Main function to run the tests.
main :: IO ()
main = do
    quickCheck prop_propagationPreservesSize
    quickCheck prop_inputSensitivity
    quickCheck prop_deterministic
    quickCheck prop_trainingReducesError
