{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}

module Main where

import Data.Csv (decodeByName, FromNamedRecord)
import Data.List (foldl', nub)
import qualified Data.ByteString.Lazy as BL
import GHC.Generics (Generic)
import qualified Matrix as M
import NN (initializeNetwork, layers, getOutput, trainEpochs, printLayerBiases, BackpropNet, trainBatch)
import ActivationFunctions

-- | Type alias for a data point, consisting of a feature vector and a label vector.
type DataPoint a = (M.ColumnVector a, M.ColumnVector a)

-- | Data type representing an Iris flower, with various features and the species.
data IrisFlower = IrisFlower {
    sepal_length :: Double,
    sepal_width :: Double,
    petal_length :: Double,
    petal_width :: Double,
    species :: String
} deriving (Show, Generic)

-- Automatically derive FromNamedRecord instances for CSV parsing
instance FromNamedRecord IrisFlower

-- | Creates a data instance from an 'IrisFlower', converting features and species to vectors.
createDataInstance :: IrisFlower -> (M.ColumnVector Double, M.ColumnVector Double)
createDataInstance iris = (featuresVector, labelVector)
  where
    features = [sepal_length iris, sepal_width iris, petal_length iris, petal_width iris]
    featuresVector = M.cvFromList features
    labelVector = encodeSpecies (species iris)

-- | Encodes the species into a one-hot vector.
encodeSpecies :: String -> M.ColumnVector Double
encodeSpecies "setosa" = M.cvFromList [1, 0, 0]
encodeSpecies "versicolor" = M.cvFromList [0, 1, 0]
encodeSpecies "virginica" = M.cvFromList [0, 0, 1]
encodeSpecies _ = error "Unknown species"

-- | Normalizes a vector based on provided minimum and range vectors.
normalizeVector :: M.ColumnVector Double -> M.ColumnVector Double -> M.ColumnVector Double -> M.ColumnVector Double
normalizeVector mins range vec = M.cvZipWith3 (\v min r -> (v - min) / r) vec mins range

-- | Applies min-max normalization to a list of column vectors.
minMaxNormalization :: [M.ColumnVector Double] -> [M.ColumnVector Double]
minMaxNormalization vectors = map (normalizeVector mins range) vectors
  where
    (mins, maxs) = getMinMax vectors
    range = M.cvZipWith (-) maxs mins

    getMinMax = foldl' updateMinMax (head vectors, head vectors) . tail
    updateMinMax (minV, maxV) vec = (M.cvZipWith min minV vec, M.cvZipWith max maxV vec)

-- | Processes CSV data to create a list of normalized data instances.
processCSV :: BL.ByteString -> [(M.ColumnVector Double, M.ColumnVector Double)]
processCSV csvData = zip normalizedFeatures labelVectors
  where
    decoded = case decodeByName csvData of
        Left err -> error ("CSV parsing error: " ++ err)
        Right (_, v) -> v
    irisFlowers = M.rvToList decoded
    dataInstances = map createDataInstance irisFlowers
    (featureVectors, labelVectors) = unzip dataInstances
    normalizedFeatures = minMaxNormalization featureVectors

-- | Tests the neural network on test data and returns the accuracy as a percentage.
testNetwork :: NN.BackpropNet -> [(M.ColumnVector Double, M.ColumnVector Double)] -> Double
testNetwork net testData = let
    outputs = map (\(features, _) -> NN.getOutput net features) testData
    successes = zipWith (\predicted (_, actual) -> if M.cvMaxIndex predicted == M.cvMaxIndex actual then 1 else 0) outputs testData
    totalSuccesses = sum successes :: Int
    totalTests = length testData :: Int
    in (fromIntegral totalSuccesses / fromIntegral totalTests) * 100


-- | Trains the neural network for a specified number of epochs and reports the training accuracy.
trainAndReport :: NN.BackpropNet -> [(M.ColumnVector Double, M.ColumnVector Double)] -> [(M.ColumnVector Double, M.ColumnVector Double)] -> Int -> IO NN.BackpropNet
trainAndReport net trainData testData numEpochs = go net numEpochs
  where
    go net 0 = return net
    go net epoch = do
      let trainedNet = trainBatch net trainData
      let accuracy = testNetwork trainedNet testData
      putStrLn $ "Epoch " ++ show (numEpochs - epoch + 1) ++ ": Training Accuracy = " ++ show accuracy
      go trainedNet (epoch - 1)

-- | Extracts the class index from a one-hot encoded label.
getClassIndex :: M.ColumnVector Double -> Int
getClassIndex = M.cvMaxIndex

-- | Extracts unique classes from the dataset.
getUniqueClasses :: [DataPoint Double] -> [Int]
getUniqueClasses dataset = nub $ map (getClassIndex . snd) dataset

-- | Filters the dataset by class index.
filterByClass :: Int -> [DataPoint Double] -> [DataPoint Double]
filterByClass idx = filter ((== idx) . getClassIndex . snd)

-- | Splits data for a single class based on the given ratio.
splitData :: Double -> [DataPoint Double] -> ([DataPoint Double], [DataPoint Double])
splitData ratio items = splitAt (floor $ fromIntegral (length items) * ratio) items

-- | Performs a stratified split of the dataset into training and test sets based on the given ratio.
stratifiedSplit :: Double -> [DataPoint Double] -> ([DataPoint Double], [DataPoint Double])
stratifiedSplit ratio dataset =
    let classes = getUniqueClasses dataset
        splitGroups = map (\cls -> splitData ratio (filterByClass cls dataset)) classes
        combineSplits f = foldl' f [] splitGroups
    in (combineSplits (\acc (train, _) -> train ++ acc),
        combineSplits (\acc (_, test) -> test ++ acc))

-- | Main function to read the CSV, initialize the network, and train the network while reporting accuracy.
main :: IO ()
main = do
    csvData <- BL.readFile "./examples/IRIS/iris.csv"
    let (trainData, testData) = stratifiedSplit 0.8 (processCSV csvData)
    -- Initialize the neural network
    let neurons = [4, 10, 3]
        activationFuncs = [sigmoidAF, sigmoidAF]
        learningRate = 0.05
        net = NN.initializeNetwork neurons activationFuncs (-0.5) 0.5 learningRate

    -- Train the network and report accuracy
    trainedNet <- trainAndReport net trainData testData 100

    {-
    -- Example usage: printing features, label, and network output for the first test data point
    let (features, label) = head testData
    putStrLn "Features matrix:"
    print features
    putStrLn "Label matrix:"
    print label
    sequence_ $ NN.printLayerBiases (NN.layers net)
    print $ NN.getOutput trainedNet features
    print $ label
    -}
    putStrLn $ "Accuracy: " ++ show (testNetwork trainedNet testData)
