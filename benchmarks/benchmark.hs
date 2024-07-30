module Main where
    
import System.Environment (getArgs)
import Text.Read (readMaybe)
import NN (initializeNetwork, trainEpochs, getOutput) -- Assuming these functions are properly defined.
import ActivationFunctions (tanhAF) -- Assuming this is defined somewhere.
import Matrix

-- | The main entry point for the program.
-- This program expects three command-line arguments:
--   1. A comma-separated list of integers representing the number of neurons in each layer of the neural network.
--   2. The batch size for training.
--   3. The number of epochs for training.
main :: IO ()
main = do
    args <- getArgs
    print args  -- Print the raw arguments to see what is received
    case args of
        [neuronsStr, batchSizeStr, epochsStr] -> do
            -- Parse the neurons string into a list of integers
            let neurons = map read . words $ map (\c -> if c == ',' then ' ' else c) neuronsStr
                -- Parse the batch size and number of epochs
                batchSize = readMaybe batchSizeStr :: Maybe Int
                epochs = readMaybe epochsStr :: Maybe Int

            case (batchSize, epochs) of
                (Just bs, Just ep) -> do
                    -- Define the activation functions, range for initial weights, and learning rate
                    let activationFuncs = replicate (length neurons - 1) tanhAF
                        a = -0.5
                        b = 0.5
                        initialNN = initializeNetwork neurons activationFuncs a b 0.1

                        -- Create dummy input and target vectors
                        input = cvReplicate (head neurons) (1 :: Double)
                        target = cvReplicate (last neurons) (1 :: Double)
                        inputs = replicate bs input
                        targets = replicate bs target

                        -- Train the neural network
                        trainedNN = trainEpochs initialNN ep (zip inputs targets)

                    -- Output the results
                    -- Uncomment the lines below if you want to print detailed results
                    -- print $ head targets
                    -- print $ mse (getOutput initialNN (head inputs)) target
                    -- print $ mse (getOutput trainedNN (head inputs)) target
                    -- Causes the network to be evaluated/trained with minimal output
                    print (cvLength (getOutput trainedNN (head inputs)) == last neurons)

                _ -> putStrLn "Invalid batch size or epochs."
        _ -> putStrLn "Usage: program <neurons> <batchSize> <epochs>"
