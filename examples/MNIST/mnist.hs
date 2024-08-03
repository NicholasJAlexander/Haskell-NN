module Main where

import qualified NN
import qualified Matrix as M
import qualified Data.ByteString as B
import Data.Word (Word8)
import ActivationFunctions

-- | Helper function to convert a 28x28 matrix to a 784x1 column vector.
flattenMatrix :: M.Matrix Double -> M.ColumnVector Double
flattenMatrix mat = M.matrixToVector $ M.transpose mat

-- | Helper function to convert a list of Word8 to a ColumnVector Double.
convertToVector :: [Word8] -> M.ColumnVector Double
convertToVector = M.cvFromList . map (\x -> fromIntegral x / 255.0)

-- | Load the MNIST images and convert them to ColumnVectors.
loadMNISTImages :: FilePath -> IO [M.ColumnVector Double]
loadMNISTImages path = do
    contents <- B.readFile path
    let images = decodeIDX3 contents
    return $ map (convertToVector . concat . chunksOf 28) images

-- | Load the MNIST labels and convert them to a list of Ints.
loadMNISTLabels :: FilePath -> IO [Int]
loadMNISTLabels path = do
    contents <- B.readFile path
    let labels = decodeIDX1 contents
    return $ map fromIntegral (B.unpack labels)

-- | Decode IDX1 format for labels.
decodeIDX1 :: B.ByteString -> B.ByteString
decodeIDX1 = B.drop 8

-- | Decode IDX3 format for images.
decodeIDX3 :: B.ByteString -> [[Word8]]
decodeIDX3 bs =
  let (_, rest) = B.splitAt 4 bs
      (_, rest2) = B.splitAt 4 rest
      (_, imageData) = B.splitAt 8 rest2
  in chunksOf (28 * 28) (B.unpack imageData)

-- | Split a list into chunks of the given size.
chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs
  | n > 0 = take n xs : chunksOf n (drop n xs)
  | otherwise = error "Chunk size must be greater than zero"

-- | Convert a label to a one-hot encoded column vector.
labelToColumnVector :: Int -> M.ColumnVector Double
labelToColumnVector label = M.cvFromList $ replicate label 0.0 ++ [1.0] ++ replicate (9 - label) 0.0

-- | Test the neural network and return the accuracy as a percentage.
testNetwork :: NN.BackpropNet -> [(M.ColumnVector Double, M.ColumnVector Double)] -> Double
testNetwork net testData = let
    outputs = map (\(features, _) -> NN.getOutput net features) testData
    successes = zipWith (\predicted (_, actual) -> if M.cvMaxIndex predicted == M.cvMaxIndex actual then 1 else 0) outputs testData
    totalSuccesses = sum successes :: Int
    totalTests = length testData :: Int
    in (fromIntegral totalSuccesses / fromIntegral totalTests) * 100

-- | Train the neural network for a specified number of epochs and report the training accuracy.
trainAndReport :: NN.BackpropNet -> [(M.ColumnVector Double, M.ColumnVector Double)] -> [(M.ColumnVector Double, M.ColumnVector Double)] -> Int -> IO NN.BackpropNet
trainAndReport net trainData testData numEpochs = go net numEpochs
  where
    go net 0 = return net
    go net epoch = do
      let trainedNet = NN.trainBatch net trainData
      let accuracy = testNetwork trainedNet testData
      putStrLn $ "Epoch " ++ show (numEpochs - epoch + 1) ++ ": Training Accuracy = " ++ show accuracy
      go trainedNet (epoch - 1)

-- | Main function to load the MNIST dataset, initialize and train the neural network, and report results.
main :: IO ()
main = do
    -- Load the MNIST dataset
    trainImages <- loadMNISTImages "./examples/MNIST/train-images-idx3-ubyte"
    trainLabels <- loadMNISTLabels "./examples/MNIST/train-labels-idx1-ubyte"
    testImages <- loadMNISTImages "./examples/MNIST/t10k-images-idx3-ubyte"
    testLabels <- loadMNISTLabels "./examples/MNIST/t10k-labels-idx1-ubyte"

    -- Initialize the neural network
    let neurons = [784, 28, 10]
        activationFuncs = [tanhAF, tanhAF]
        learningRate = 0.05
        net = NN.initializeNetwork neurons activationFuncs (-0.5) 0.5 learningRate

    -- Train the network
    let trainData = zip trainImages (map labelToColumnVector trainLabels)
        testData = zip testImages (map labelToColumnVector testLabels)
    
    trainedNet <- trainAndReport net trainData testData 20
    
    print $ NN.getOutput trainedNet (head trainImages)
    print $ head trainLabels

    -- Uncomment the below code to evaluate the network on the test set
    -- let accuracy = testNetwork trainedNet testData
    -- putStrLn $ "Test accuracy: " ++ show accuracy
