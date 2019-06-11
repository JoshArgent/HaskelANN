module DataProcessing where

import Data.List.Split
import Data.List
import AutomaticDifferentiation
import System.Random

-- A sample is a tuple of features and labels
type Sample = ([[C]], [[C]])

        
-- Load a CSV data file and prepare the data for training (apply shuffle, normalisation)
-- Data CSV file -> No. Features to extract -> (Features, Labels)
loadData :: FilePath -> Int -> IO Sample
loadData f a = do
                contents <- readFile f
                let rows = lines contents
                let entires = map (splitOn ",") rows
                let values = map (map (constC . read)) entires
                values <- shuffle values
                let normalised = normalise values
                let features = map (take a) normalised
                let labels = map (drop a) normalised
                return (features, labels)

                
-- Split data sample into two sets based on a given fraction (set1, set2)
splitData :: Sample -> Double -> (Sample, Sample)
splitData (features, labels) x = ((drop amount features, drop amount labels), (take amount features, take amount labels))
                        where
                                amount = round (x * (fromIntegral $ length features)) :: Int


-- Randomly shuffle a list                                
-- Credit: https://www.programming-idioms.org/idiom/10/shuffle-a-list/826/haskell                
shuffle :: [a] -> IO [a]                
shuffle x = if length x < 2 then return x else do
        i <- System.Random.randomRIO (0, length(x)-1)
        r <- shuffle (take i x ++ drop (i+1) x)
        return (x!!i : r)    


-- Normalise each feature or label to be within the range 0 to 1
-- (This make ANN training easier - less likely to get stuck in a local minimum)        
normalise :: [[C]] -> [[C]]
normalise xs = transpose $ map norm $ transpose xs
        where
            norm ys = norm' (minimum ys) (maximum ys) ys  
            norm' _ _ [] = []
            norm' minY maxY (y:ys) = (((y - minY) / (maxY - minY)) : (norm' minY maxY ys))      
                
           
-- Determines the integer class of a classification result
-- E.g. [1, 0, 0] = 1     [0, 0, 1] = 3        
classOf :: [C] -> Int
classOf [] = 1
classOf (x:xs) = 1 + if (round $ valC x) == 1 then 0 else classOf xs

        
                