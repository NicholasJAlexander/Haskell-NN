{-# LANGUAGE FlexibleContexts #-}
{-# OPTIONS_GHC -Wno-simplifiable-class-constraints #-}
module Matrix where

import qualified Data.Vector as V
import qualified Data.Vector.Unboxed as UV

-- | Type alias for a matrix, represented as a vector of unboxed vectors.
type Matrix a = V.Vector (UV.Vector a)

-- | Type alias for a column vector, represented as an unboxed vector.
type ColumnVector a = UV.Vector a

-- * Matrix Functions

-- | Converts a list of lists to a matrix (vector of vectors).
-- This function checks for uniformity of inner list lengths to ensure a valid matrix.
fromLists :: UV.Unbox a => [[a]] -> Matrix a
fromLists xss
    | all (== head lengths) lengths = V.fromList . map UV.fromList $ xss
    | otherwise = error "All rows must have the same number of columns"
  where 
    lengths = map length xss

toList :: UV.Unbox a => Matrix a -> [a]
toList = V.foldr ((++) . UV.toList) []

-- | Returns the number of columns in the matrix.
ncols :: UV.Unbox a => Matrix a -> Int
ncols m = if V.null m then 0 else UV.length (V.head m)

-- | Returns the number of rows in the matrix.
nrows :: UV.Unbox a => Matrix a -> Int
nrows = V.length

-- | Calculates the Hadamard product (element-wise multiplication) of two matrices.
-- This function ensures that matrices are of the same dimensions.
hadamardProduct :: (Floating a, UV.Unbox a) => Matrix a -> Matrix a -> Matrix a
hadamardProduct m1 m2
    | nrows m1 /= nrows m2 || ncols m1 /= ncols m2 = error "Matrices must have the same dimensions for Hadamard product"
    | otherwise = V.zipWith (UV.zipWith (*)) m1 m2

-- | Prints a matrix.
printMatrix :: (Show a, UV.Unbox a) => Matrix a -> IO ()
printMatrix matrix = do
    mapM_ printRow (V.toList matrix)
    putStrLn ""  -- Add an extra newline at the end
  where
    printRow row = print (UV.toList row)

-- | Scales a matrix by a scalar value.
scaleMatrix :: (UV.Unbox a, Num a) => a -> Matrix a -> Matrix a
scaleMatrix scalar = V.map (UV.map (* scalar))

-- | Applies an element-wise operation to two matrices.
elementwise :: (UV.Unbox a) => (a -> a -> a) -> Matrix a -> Matrix a -> Matrix a
elementwise f = V.zipWith (UV.zipWith f)

-- | Standard matrix multiplication.
-- Multiplies two matrices using the standard matrix multiplication algorithm.
multStd :: (Num a, UV.Unbox a) => Matrix a -> Matrix a -> Matrix a
multStd m1 m2
    | ncols m1 /= nrows m2 = error "Number of columns in the first matrix must equal number of rows in the second"
    | otherwise = V.generate (nrows m1) $ \i ->
        UV.generate (ncols m2) $ \j ->
            UV.sum $ UV.zipWith (*) (m1 V.! i) (UV.generate (nrows m2) $ \k -> (m2 V.! k) UV.! j)

-- | Transposes a matrix, converting rows into columns.
transpose :: UV.Unbox a => Matrix a -> Matrix a
transpose m
    | V.null m = V.empty
    | otherwise = V.generate (ncols m) $ \i ->
        UV.generate (nrows m) $ \j -> (m V.! j) UV.! i

-- | Converts a column vector into a single-column matrix where each element of the vector becomes a separate row.
vectorToMatrix :: UV.Unbox a => UV.Vector a -> Matrix a
vectorToMatrix v = V.generate (UV.length v) (\i -> UV.singleton (v UV.! i))

-- | Converts a single-column matrix back into a column vector.
-- Assumes the matrix has exactly one column.
matrixToVector :: UV.Unbox a => Matrix a -> UV.Vector a
matrixToVector m = UV.concat $ V.toList m

-- | Multiplies a matrix by a column vector.
matrixVectorMult :: (Num a, UV.Unbox a) => Matrix a -> ColumnVector a -> ColumnVector a
matrixVectorMult m v
    | ncols m /= UV.length v = error "Dimension mismatch: Matrix columns must equal vector length"
    | otherwise = UV.generate (nrows m) generateElement
  where
    generateElement i = UV.sum $ UV.zipWith (*) (m V.! i) v

-- | Custom matrix-vector multiplication, combining element-wise operations on the vector and matrix elements.
customMatrixVectorMult :: (Num a, UV.Unbox a) => Matrix a -> UV.Vector a -> UV.Vector a -> UV.Vector a
customMatrixVectorMult wKT dazzleK f'aK
    | ncols wKT /= UV.length dazzleK = error "Dimension mismatch: Matrix columns must equal length of vectors"
    | otherwise = UV.fromList $ V.toList $ V.map (\row ->
        UV.sum $ UV.zipWith (*) row (UV.zipWith (*) dazzleK f'aK)) wKT

-- | Returns the dimensions of a matrix as a tuple (number of rows, number of columns).
dimensions :: Matrix Double -> (Int, Int)
dimensions matrix = 
    if V.null matrix
    then (0, 0)
    else (V.length matrix, UV.length (V.head matrix))

-- | Applies a function to each element of a column vector.
cvMap :: (UV.Unbox a, UV.Unbox b) => (a -> b) -> ColumnVector a -> ColumnVector b
cvMap = UV.map

-- | Sums all elements of a column vector.
cvSum :: (UV.Unbox Double) => ColumnVector Double -> Double
cvSum = UV.sum

-- | Converts a column vector to a list.
cvToList :: (UV.Unbox a) => ColumnVector a -> [a]
cvToList = UV.toList

-- | Checks if all elements in a column vector satisfy a predicate.
cvAll :: UV.Unbox a => (a -> Bool) -> UV.Vector a -> Bool
cvAll = UV.all

-- | Returns the length of a column vector.
cvLength :: UV.Unbox a => ColumnVector a -> Int
cvLength = UV.length

-- | Generates a column vector of a given length, applying a function to each index.
cvGenerate :: UV.Unbox a => Int -> (Int -> a) -> ColumnVector a
cvGenerate = UV.generate

-- | Generates a matrix of a given number of rows, applying a function to each row index.
generateMatrix :: Int -> (Int -> ColumnVector a) -> Matrix a
generateMatrix = V.generate

-- | Retrieves an element from a column vector by index.
cvGetElem :: UV.Unbox a => ColumnVector a -> Int -> a
cvGetElem = (UV.!)

-- | Applies a binary function element-wise to two column vectors.
cvZipWith :: (UV.Unbox a) => (a -> a -> a) -> ColumnVector a -> ColumnVector a -> ColumnVector a
cvZipWith = UV.zipWith

-- | Applies a ternary function element-wise to three column vectors.
cvZipWith3 :: (UV.Unbox a) => (a -> a -> a -> a) -> ColumnVector a -> ColumnVector a -> ColumnVector a -> ColumnVector a
cvZipWith3 = UV.zipWith3

-- | Creates a column vector of a given length, with all elements initialized to a given value.
cvReplicate :: (UV.Unbox a) => Int -> a -> ColumnVector a
cvReplicate = UV.replicate

-- | Converts a list to a column vector.
cvFromList :: (UV.Unbox a) => [a] -> ColumnVector a
cvFromList = UV.fromList

-- | Adds two column vectors element-wise.
cvAdd :: (Num a, UV.Unbox a) => ColumnVector a -> ColumnVector a -> ColumnVector a
cvAdd = UV.zipWith (+)

-- | Finds the maximum value in a column vector.
cvMaximum :: (UV.Unbox a, Eq a, Ord a) => ColumnVector a -> a
cvMaximum = UV.maximum

-- | Converts a matrix to a list of lists.
mToList :: UV.Unbox a => Matrix a -> [[a]]
mToList matrix = map UV.toList (V.toList matrix)

-- | Compares if two matrices are the same.
-- This function first checks if the dimensions of the two matrices are the same,
-- and then it checks if all corresponding elements are equal.
equalMatrices :: (Eq a, UV.Unbox a) => Matrix a -> Matrix a -> Bool
equalMatrices m1 m2
    | nrows m1 /= nrows m2 || ncols m1 /= ncols m2 = False
    | otherwise = V.and $ V.zipWith (\row1 row2 -> UV.and $ UV.zipWith (==) row1 row2) m1 m2
    
-- | Replicates a monadic action a given number of times, collecting the results into an unboxed vector.
cvReplicateM :: (Monad m, UV.Unbox a) => Int -> m a -> m (UV.Vector a)
cvReplicateM = UV.replicateM

-- | Finds the index of the maximum element in a column vector.
cvMaxIndex :: (UV.Unbox a, Ord a) => ColumnVector a -> Int 
cvMaxIndex = UV.maxIndex

-- | Converts a row vector to a list.
rvToList :: V.Vector a -> [a]
rvToList = V.toList

-- | Calculated difference between two matrices
matrixDifference :: (UV.Unbox a, Num a) => Matrix a -> Matrix a -> [a]
matrixDifference m1 m2 = map abs (zipWith (-) (toList m1) (toList m2))