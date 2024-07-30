{-# LANGUAGE InstanceSigs #-}
module ActivationFunctions (
    ActivationFunc(..),
    tanhAF,
    sigmoidAF,
    reluAF,
    leakyReluAF,
    eluAF
) where

import Matrix

-- | Data type representing an activation function in a neural network.
--   Contains the function, its derivative, and a description.
data ActivationFunc = ActivationFunc {
    -- | The activation function itself. 
    --   Takes a 'ColumnVector Double' and returns a 'ColumnVector Double'.
    aF :: ColumnVector Double -> ColumnVector Double,

    -- | The derivative of the activation function.
    --   This is necessary for backpropagation in neural networks.
    aF' :: ColumnVector Double -> ColumnVector Double,

    -- | A descriptive string for the activation function.
    desc :: String
}

-- | Tanh activation function.
tanhAF :: ActivationFunc
tanhAF = ActivationFunc {
    aF = cvMap tanh,
    aF' = cvMap $ \x -> 1 - tanh x ^ (2 :: Int),
    desc = "Tanh"
}

-- | Sigmoid activation function.
sigmoidAF :: ActivationFunc
sigmoidAF = ActivationFunc {
    aF = cvMap $ \x -> 1 / (1 + exp (-x)),  -- The sigmoid function
    aF' = cvMap (\x -> let sig = 1 / (1 + exp (-x)) in sig * (1 - sig)),  -- The derivative of the sigmoid function
    desc = "Sigmoid"
}

-- | ReLU activation function.
reluAF :: ActivationFunc
reluAF = ActivationFunc {
    aF = cvMap $ \x -> max 0 x,
    aF' = cvMap $ \x -> if x > 0 then 1 else 0,
    desc = "ReLU"
}

-- | Leaky ReLU activation function.
leakyReluAF :: ActivationFunc
leakyReluAF = ActivationFunc {
    aF = cvMap $ \x -> if x > 0 then x else 0.01 * x,
    aF' = cvMap $ \x -> if x > 0 then 1 else 0.01,
    desc = "Leaky ReLU"
}

-- | ELU activation function.
eluAF :: ActivationFunc
eluAF = ActivationFunc {
    aF = cvMap $ \x -> if x >= 0 then x else (exp x - 1),
    aF' = cvMap $ \x -> if x >= 0 then 1 else exp x,
    desc = "ELU"
}

-- | Custom 'Show' instance for 'ActivationFunc' to display the description.
instance Show ActivationFunc where
    show :: ActivationFunc -> String
    show = desc
