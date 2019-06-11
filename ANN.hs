module ANN where

import AutomaticDifferentiation
import DataProcessing

import Data.List
import Control.Monad
import System.Random


{-
Network Architecture
-}

-- Used to differentiate between what value is a weight and a bias
type Weight = C
type Bias = Weight

-- A list of lists of weights for each output neuron of the fully-connected layer
-- Eg. If the layer has 2 inputs and 3 outputs, then this could be a valid layer def:
-- Layer [0,0,0] [[1,1], [1,1], [1,1]] sigmoid
data Layer = Layer [Bias] [[Weight]] ActivationFunction 

instance Show Layer where
        show (Layer bs wss _) = (show bs) ++ ", " ++ (show wss)

-- All the layers of the ANN, in order
type NeuralNetwork = [Layer]

-- The ANN Hyperparams
type LearningRate = Double
type Epochs = Int
data Params = Params LearningRate ErrorFunction Epochs

-- Build a NeuralNetwork with a given number of layers and units
-- Randomly initialises the weights and biases
buildNetwork :: [Int] -> [ActivationFunction] -> IO NeuralNetwork
buildNetwork (ins:outs:xs) (af:afs) = do 
                biases <- randomList outs
                weights <- replicateM outs (randomList ins)
                let layer = Layer (map dVarC biases) (map (map dVarC) weights) af
                nextLayers <- buildNetwork (outs:xs) afs
                return (layer : nextLayers)
buildNetwork _ _ = return []

-- Generate a list of a given length containing random numbers
randomList :: Int -> IO([Double])
randomList n = replicateM n $ randomIO



{-
Error Functions
-}
type ErrorFunction = (C -> C -> C)

-- Squared Error                        
squareError :: ErrorFunction                        
squareError x y = (x - y) * (x - y)

-- Cross Entropy Error (Multiple Independant Binary Classification)
crossEntropyError :: ErrorFunction
crossEntropyError y t = negate (t * (log y) + ((1 - t) * (log (1 - y))))                  
                                
-- Classification error, 1 if correct classification, 0 otherwise                                
classification :: ErrorFunction
classification y t | (round $ valC y) == (round $ valC t) = 1
                   | otherwise = 0

-- Apply an error function to a whole set of predicted labels and target labels                                
applyErrorFunction :: ErrorFunction -> [[C]] -> [[C]] -> C
applyErrorFunction f y t = sum $ map sum (zipWith (zipWith f) y t)



{-
Activation Functions
-}
type ActivationFunction = (C -> C)

-- Rectified Linear Unit (ReLU Softplus) activation function 
relu :: ActivationFunction
relu x = log (1 + (exp x))

-- Sigmoid activation function
sigmoid :: ActivationFunction
sigmoid x = 1.0 / (1.0 + (exp (-x)))

-- Linear activation function
linear :: ActivationFunction
linear = id

-- Calculate the activation output for a particular neuron 
-- Uses 'dVarC $ valC' so that only the activation function, f, is differentiatiable
activation :: [C] -> [Weight] -> Bias -> ActivationFunction -> C
activation xs weights b f = f $ dVarC $ valC $ (sum $ zipWith (*) xs weights) + b                               
                                

                                
{-
Training
-}                                
                                
-- Forward propogate an input(s) through a given ANN
-- Returns a list of neuron activations for each layer
forwardPropagate :: NeuralNetwork -> [C] -> [[C]]
forwardPropagate [] _ = []
forwardPropagate _ [] = []
forwardPropagate ((Layer bs wss af):ls) xs = (output : (forwardPropagate ls output))
                where
                        wbss = zip bs wss
                        output = map (\(b, ws) -> activation xs ws b af) wbss

                                         
{-
Backpropagation wrapper, reverses the layers and activations so that it can work backwards easily  
nn - neural network to train
xss - forward propagation activations
ys - training data labels
ins - training data features
ef - error function
lr - learning rate
-}             
backPropagate ::  NeuralNetwork -> [[C]] -> [C] -> [C] -> ErrorFunction -> LearningRate -> NeuralNetwork
backPropagate nn xss ins ys ef lr = tail $ reverse $ backPropagate' (reverse nn) (reverse (ins:xss)) ys ef lr


{-
Backpropagate the output layer

bs - list of biases for a layer
wss - matrix of weights for a layer
ls - list of preceding layers
xs1 - neuron outputs for current layer
xs2 - neuron inputs for current layer
ys - expected neuron outputs for current layer
lr - learning rate
z:zs = xs2

propNeuron - calculate new incoming weight values for a neuron
newWeights - calculate new weight values for this layer
es - total error w.r.t the output of each neuron
cs - neuron outputs w.r.t the total neuron inputs
es and cs are used in next layer of back propogation
-}
backPropagate' :: NeuralNetwork -> [[C]] -> [C] -> ErrorFunction -> LearningRate -> NeuralNetwork
backPropagate' ((Layer bs wss af):ls) (xs1:xs2:xss) ys ef lr = ((Layer bs newWeights af) : (backPropagateHidden' ls (xs2:xss) es cs wss lr))
        where
                propNeuron (z:zs) x1 y1 (w:ws) = ((updateWeight x1 y1 z w ef lr) : propNeuron zs x1 y1 ws)
                propNeuron [] _ _ _ = []
                propNeuron _ _ _ [] = []
                result = zipWith3 (propNeuron xs2) xs1 ys wss
                newWeights = map (\x -> map fst x) result
                es = map (\x -> map (fst . snd) x) result
                cs = map (\x -> map (snd . snd) x) result


{-
Backpropagate the hidden layer(s)

bs - list of biases for a layer
wss - matrix of weights for a layer
ls - list of preceding layers
xs1 - neuron outputs for current layer
xs2 - neuron inputs for current layer
es - total error w.r.t the output of each neuron from previous layer (i.e. one layer ahead)
ds - neuron outputs w.r.t the total neuron inputs from previous layer
cs - the old weights of the previous layer
lr - learning rate

es, ds, cs are transposed so that the lists show errors/weights from an outgoing neuron as opposed to an incoming neuron
e.g. [[w5, w6], [w7, w8]] => [[w5, w7], [w6, w8]]

propNeuron - calculate new incoming weight values for a neuron
newWeights - calculate new weight values for this layer
es3 - total error w.r.t the output of each neuron
ds3 - neuron outputs w.r.t the total neuron inputs
-}
backPropagateHidden' :: NeuralNetwork -> [[C]] -> [[C]] -> [[C]] -> [[Weight]] -> LearningRate -> NeuralNetwork
backPropagateHidden' [] _ _ _ _ _ = [(Layer [] [] linear)]
backPropagateHidden' ((Layer bs wss af):ls) (xs1:xs2:xss) es ds cs lr = (Layer bs newWeights af) : (backPropagateHidden' ls (xs2:xss) es3 ds3 wss lr)
        where
                propNeuron (z:inputs) es2 ds2 outWeights2 x1 (w:ws) = 
                        ((updateWeightHidden es2 ds2 outWeights2 x1 z w lr) : propNeuron inputs es2 ds2 outWeights2 x1 ws)
                propNeuron [] _ _ _ _ _ = []  
                propNeuron _ _ _ _ _ [] = []                                          
                result = zipWith5 (propNeuron xs2) (transpose es) (transpose ds) (transpose cs) xs1 wss
                newWeights = map (\x -> map fst x) result
                es3 = map (\x -> map (fst . snd) x) result
                ds3 = map (\x -> map (snd . snd) x) result

                
{-
predicted - the output value of the output neuron
expected - the expected/target value of the output neuron
input - the input to the output neuron
weight - the weight to be updated
lr - the learning rate

a - total error change w.r.t the neuron output
b - output change w.r.t the neuron input
c - input change w.r.t the current weight (= the input value)
a * b * c - derivative of total error w.r.t the weight
Return: (d, (a, b)) [a and b used in subsequent calculations]
d - new weight value
-}                                        
updateWeight :: C -> C -> C -> C -> ErrorFunction -> LearningRate -> (C, (C, C))
updateWeight predicted expected input weight ef lr = (dVarC ((valC weight) - (lr * a * b * c)), (dVarC a, dVarC b))
                where
                      a = valC $ derC $ ef (dVarC $ valC $ predicted) expected
                      b = valC $ derC $ predicted
                      c = valC $ input    
      

{-
a - total error change w.r.t the hidden neuron's output
b - output change w.r.t the neuron input
c - input change w.r.t the current weight (= the input value)
a * b * c - derivative of total error w.r.t the weight
Return: (d, (a, b)) [a and b used in subsequent calculations]
d - new weight value
-}                      
updateWeightHidden :: [C] -> [C] -> [C] -> C -> C -> C -> LearningRate -> (C, (C, C))
updateWeightHidden errors outs outWeights output input weight lr = (dVarC ((valC weight) - (lr * a * b * c)), (dVarC a, dVarC b))
                where
                      a = valC $ sum $ zipWith (*) outWeights $ zipWith (*) errors outs
                      b = valC $ derC $ output
                      c = valC $ input

                      
                      
-- Wrapper for train' function                      
train :: NeuralNetwork -> Params -> Sample -> IO NeuralNetwork
train nn ps s = train' nn ps s 1 99999 nn                   
                      
                      
{-
Train an ANN on a set of inputs and targets for a given number of epochs
-}                      
train' :: NeuralNetwork -> Params -> Sample -> Int -> Double -> NeuralNetwork -> IO NeuralNetwork
train' nn (Params _ _ 0) _ _ _ _ = return nn     
train' nn ps@(Params lr ef epochs) s epoch prevValError best =  do
                        let ((tfs, tls), (vfs, vls))    = splitData s 0.2  -- 20% of data used for validation
                        let trainedNN                   = last $ scanl (trainSample ps) nn (zip tfs tls)
                        let nextNN                      = train' trainedNN (Params lr ef (epochs - 1)) s (epoch + 1)
                        let trainError                  = (valC $ validate trainedNN ef (tfs, tls)) / (fromIntegral $ length tfs)
                        let valError                    = (valC $ validate trainedNN ef (vfs, vls)) / (fromIntegral $ length vfs)
                        putStrLn $ "Epoch #" ++ (show epoch) ++ ", T. err = " ++ (show trainError) ++ ", V. err = " ++ (show valError)
                        if epoch `mod` 8 == 0 then
                                do
                                        if valError >= prevValError then do
                                                putStrLn "Training stopped!"
                                                return best
                                        else
                                                nextNN valError trainedNN
                        else
                                nextNN prevValError best                                                          
                        
                   

{-
Train an individual sample using backpropogation and return a new ANN
-}                        
trainSample :: Params -> NeuralNetwork -> ([C], [C]) -> NeuralNetwork
trainSample (Params lr ef _) nn (xs, ys) = backPropagate nn fwd xs ys ef lr
                where
                        fwd = forwardPropagate nn xs


{-
Return the outputs for a given input sample
-}
sim :: NeuralNetwork -> Sample -> [[C]]
sim nn (features, _) = map last (map (forwardPropagate nn) features)


{-
Calculate the validation/test error for a given sample of data and a trained NeuralNetwork
-}
validate :: NeuralNetwork -> ErrorFunction -> Sample -> C
validate nn ef s@(features, labels) = applyErrorFunction ef (sim nn s) labels
                
