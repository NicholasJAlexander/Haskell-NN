{-# OPTIONS_GHC -Wno-deprecations #-}
{-# LANGUAGE FlexibleInstances #-}

module Main where

import Test.QuickCheck
import System.Exit (exitFailure, exitSuccess)
import Control.Monad (unless)
import NN
import Matrix
import ActivationFunctions (ActivationFunc(..), tanhAF, sigmoidAF)
import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV

-- * Newtype Wrappers

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

-- * Error Measures

-- | Mean squared error. Strictly non-negative -- a test that compares errors is
--   only meaningful if this is, so the sign here is load-bearing.
mse :: ColumnVector Double -> ColumnVector Double -> Double
mse output target =
    let diff = cvZipWith (-) output target
        squared = cvMap (\x -> x * x) diff
    in cvSum squared / fromIntegral (cvLength output)

-- | The error the network's backpropagation actually descends: @E = 0.5 * sum (y - t)^2@.
--   'mseGrad' is by definition its derivative with respect to @y@, so this is the
--   function the analytic gradients must agree with.
sumSquaredError :: BackpropNet -> ColumnVector Double -> ColumnVector Double -> Double
sumSquaredError net input target =
    0.5 * cvSum (cvMap (\x -> x * x) (cvZipWith (-) (getOutput net input) target))

-- * Generators

-- | Generator for a list of neurons (layer sizes).
genNeurons :: Gen [Int]
genNeurons = do
  numLayers <- choose (3, 5)
  vectorOf numLayers (choose (1, 10))

-- | Generator for a 'BackpropNet'.
genBackpropNet :: Gen TestBackpropNet
genBackpropNet = do
  neurons <- genNeurons
  let activationFuncs = replicate (length neurons - 1) tanhAF
  a <- choose (-0.1, 0)
  b <- choose (0, 0.1)
  lRate <- choose (0.05, 0.1)
  return $ TestBackpropNet $ initializeNetwork neurons activationFuncs a b lRate

-- | A deliberately small network, over both activation functions, for the
--   gradient check -- which costs two forward passes per parameter.
genSmallNet :: Gen TestBackpropNet
genSmallNet = do
  numLayers <- choose (2, 3)
  neurons <- vectorOf numLayers (choose (2, 4))
  af <- elements [sigmoidAF, tanhAF]
  let activationFuncs = replicate (length neurons - 1) af
  return $ TestBackpropNet $ initializeNetwork neurons activationFuncs (-0.6) 0.6 0.1

-- | Generator for input-target pairs matching a network's shape.
genInputTargetPair :: BackpropNet -> Gen (ColumnVector Double, ColumnVector Double)
genInputTargetPair net = do
    let inputSize = ncols . lWeights . head $ layers net
        outputSize = nrows . lWeights . last $ layers net
    input <- vectorOf inputSize (choose (-1, 1))
    targetIndex <- choose (0, outputSize - 1)
    let target = cvFromList $ replicate targetIndex 0 ++ [1] ++ replicate (outputSize - targetIndex - 1) 0
    return (cvFromList input, target)

-- * Parameter Surgery
--
--   Helpers for perturbing a single weight or bias, so the analytic gradients can
--   be checked against numerical ones.

mapLayer :: Int -> (Layer -> Layer) -> BackpropNet -> BackpropNet
mapLayer l f net = net { layers = [ if k == l then f lay else lay
                                  | (k, lay) <- zip [0 ..] (layers net) ] }

setWeight :: Int -> Int -> Int -> Double -> BackpropNet -> BackpropNet
setWeight l i j v = mapLayer l $ \lay ->
    let m = lWeights lay in lay { lWeights = m V.// [(i, (m V.! i) UV.// [(j, v)])] }

getWeight :: Int -> Int -> Int -> BackpropNet -> Double
getWeight l i j net = (lWeights (layers net !! l) V.! i) UV.! j

setBias :: Int -> Int -> Double -> BackpropNet -> BackpropNet
setBias l i v = mapLayer l $ \lay -> lay { lBiases = lBiases lay UV.// [(i, v)] }

getBias :: Int -> Int -> BackpropNet -> Double
getBias l i net = lBiases (layers net !! l) UV.! i

-- | Central difference of the error with respect to one parameter.
centralDifference :: BackpropNet
                  -> ColumnVector Double -> ColumnVector Double
                  -> (Double -> BackpropNet)  -- ^ rebuild the net with the parameter set to a value
                  -> Double                   -- ^ the parameter's current value
                  -> Double
centralDifference _ input target rebuild v =
    (sumSquaredError (rebuild (v + h)) input target -
     sumSquaredError (rebuild (v - h)) input target) / (2 * h)
  where h = 1e-6

-- | Analytic and numerical gradients agree if they are close in absolute terms,
--   or -- for larger gradients, where finite differences lose precision -- in
--   relative terms.
closeEnough :: Double -> Double -> Bool
closeEnough analytic numeric =
    absErr < 1e-7 || absErr / max 1e-12 (abs analytic + abs numeric) < 1e-5
  where absErr = abs (analytic - numeric)

-- * Property Tests

-- | The analytic weight gradients must agree with central differences of the very
--   error function the network descends. This is the regression test for the
--   backpropagation chain rule.
weightGradientsAreCorrect :: Property
weightGradientsAreCorrect =
  withMaxSuccess 30 $
  forAll genSmallNet $ \(TestBackpropNet net) ->
  forAll (genInputTargetPair net) $ \(input, target) ->
    let bps = backpropagateNet target (propagateNet input net)
        checks =
          [ (l, i, j, analytic, numeric)
          | (l, bp) <- zip [0 ..] bps
          , let g = bpErrGrad bp
          , i <- [0 .. nrows g - 1]
          , j <- [0 .. ncols g - 1]
          , let analytic = (g V.! i) UV.! j
                numeric  = centralDifference net input target
                             (\v -> setWeight l i j v net) (getWeight l i j net)
          ]
        bad = [ c | c@(_, _, _, a, n) <- checks, not (closeEnough a n) ]
    in counterexample ("mismatched weight gradients: " ++ show bad) (null bad)

-- | The analytic bias gradients must agree with central differences too.
--
--   This is the property that would have caught the missing f'(a) factor: a bias
--   gradient read off dE\/dy rather than dE\/da is wrong by exactly 1\/f'(a),
--   which for a sigmoid near the origin is a factor of four.
biasGradientsAreCorrect :: Property
biasGradientsAreCorrect =
  withMaxSuccess 30 $
  forAll genSmallNet $ \(TestBackpropNet net) ->
  forAll (genInputTargetPair net) $ \(input, target) ->
    let bps = backpropagateNet target (propagateNet input net)
        checks =
          [ (l, i, analytic, numeric)
          | (l, bp) <- zip [0 ..] bps
          , let g = bpBiasGrad bp
          , i <- [0 .. cvLength g - 1]
          , let analytic = g UV.! i
                numeric  = centralDifference net input target
                             (\v -> setBias l i v net) (getBias l i net)
          ]
        bad = [ c | c@(_, _, a, n) <- checks, not (closeEnough a n) ]
    in counterexample ("mismatched bias gradients: " ++ show bad) (null bad)

-- | A training step must actually descend: biases carry their previous value into
--   the update rather than being overwritten by the gradient.
--
--   This is the regression test for the bias update. A rule that discards the
--   incoming bias produces a step that does not track the error downhill.
trainingReducesError :: Property
trainingReducesError =
  withMaxSuccess 50 $
  forAll genSmallNet $ \(TestBackpropNet net0) ->
  forAll (genInputTargetPair net0) $ \(input, target) ->
    let net = net0 { learningRate = 0.05 }
        before = mse (getOutput net input) target
        after  = mse (getOutput (trainSingleExample net (input, target)) input) target
    in counterexample ("error before: " ++ show before ++ "\nerror after:  " ++ show after)
       (after <= before)

-- | Biases must accumulate across steps rather than being replaced each time.
--   Training twice on the same example must move a bias strictly further than
--   training once -- an overwriting update lands in the same place both times.
biasesAccumulate :: Property
biasesAccumulate =
  withMaxSuccess 50 $
  forAll genSmallNet $ \(TestBackpropNet net0) ->
  forAll (genInputTargetPair net0) $ \(input, target) ->
    let net   = net0 { learningRate = 0.2 }
        once  = trainSingleExample net (input, target)
        twice = trainSingleExample once (input, target)
        b0 = lBiases (last (layers net))
        b1 = lBiases (last (layers once))
        b2 = lBiases (last (layers twice))
        moved v w = cvSum (cvMap abs (cvZipWith (-) v w))
    in counterexample ("step 1 moved " ++ show (moved b1 b0) ++
                       ", step 2 moved " ++ show (moved b2 b1))
       (moved b2 b0 > moved b1 b0)

-- | Propagation produces an output of the last layer's size.
propagationPreservesSize :: Property
propagationPreservesSize =
  forAll genBackpropNet $ \(TestBackpropNet net) ->
  forAll (vectorOf (inputSize net) (choose (0.1, 0.5))) $ \input ->
    cvLength (getOutput net (cvFromList input)) == lastLayerSize net
  where
    inputSize net = ncols . lWeights . head $ layers net
    lastLayerSize net = nrows . lWeights . last $ layers net

-- | Small input changes produce small output changes.
inputSensitivity :: Property
inputSensitivity =
  forAll genBackpropNet $ \(TestBackpropNet net) ->
  forAll (genInputTargetPair net) $ \(input, _) ->
    let output1 = getOutput net input
        output2 = getOutput net (cvMap (+ 1e-5) input)
    in mse output1 output2 < 0.01

-- | The network is a pure function of its input.
isDeterministic :: Property
isDeterministic =
  forAll genBackpropNet $ \(TestBackpropNet net) ->
  forAll (genInputTargetPair net) $ \(input, _) ->
    cvToList (getOutput net input) == cvToList (getOutput net input)

-- * Runner

-- | Runs a named property, reporting pass or failure, and returns whether it passed.
runNamed :: String -> Property -> IO Bool
runNamed name prop = do
    putStrLn ("=== " ++ name)
    result <- quickCheckResult prop
    return (isSuccess result)

main :: IO ()
main = do
    results <- mapM (uncurry runNamed)
      [ ("weightGradientsAreCorrect", weightGradientsAreCorrect)
      , ("biasGradientsAreCorrect",   biasGradientsAreCorrect)
      , ("trainingReducesError",      trainingReducesError)
      , ("biasesAccumulate",          biasesAccumulate)
      , ("propagationPreservesSize",  propagationPreservesSize)
      , ("inputSensitivity",          inputSensitivity)
      , ("isDeterministic",           isDeterministic)
      ]
    unless (and results) exitFailure
    exitSuccess
