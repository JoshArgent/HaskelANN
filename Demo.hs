module Demo where

import ANN
import DataProcessing
import AutomaticDifferentiation

{-
Demonstrates a multi-class classification problem of species of iris plants from leaf/stem measurements.
-}
demo1 :: IO ()
demo1 = do
        putStrLn "Iris classification problem:"
        putStrLn "Using 80% training and 20% test data..."
        sample <- loadData "iris.csv" 4 -- Load the data with 4 features
        let (trainingData, testData) = splitData sample 0.2 -- Split the data into 80% training and 20% testing
        nn <- buildNetwork [4, 5, 5, 3] [relu, sigmoid, sigmoid] -- Build a 2 hidden layer ANN with relu and sigmoid activations
        let params = Params 0.5 squareError 100 -- Use a learning rate of 0.5, square error function and 100 epochs
        trainedNN <- train nn params trainingData -- Train the ANN
        let result = (validate trainedNN classification testData) / (fromIntegral $ (length $ snd testData) * (fromIntegral $ (length $ head $ snd testData)))
        putStrLn $ "Correct Classification Rate (using test data) = " ++ (show $ valC result) -- Show the test error
        return ()


{-
Demonstrates a binary classification problem of determining whether a cell is cancerous of not
-}        
demo2 :: IO ()
demo2 = do
        putStrLn "Breast Cancer classification problem:"
        putStrLn "Using 80% training and 20% test data..."
        sample <- loadData "breast_cancer.csv" 30 -- Load the data with 30 features
        let (trainingData, testData) = splitData sample 0.2 -- Split into 80% training and 20% testing
        nn <- buildNetwork [30, 14, 2] [sigmoid, sigmoid] -- Build a 1 hidden layer ANN with sigmoid activations
        let params = Params 0.1 crossEntropyError 100 -- Use a learning rate of 0.1, cross entropy error function and 100 epochs
        trainedNN <- train nn params trainingData -- Train the ANN
        let result = (validate trainedNN classification testData) / (fromIntegral $ (length $ snd testData) * (fromIntegral $ (length $ head $ snd testData)))
        putStrLn $ "Correct Classification Rate (using test data) = " ++ (show $ valC result) -- Show the test error
        return ()            
         
        