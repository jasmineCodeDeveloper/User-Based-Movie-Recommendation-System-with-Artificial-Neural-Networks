from preprocess import Preprocess
from neural_network import NeuralNetwork
import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Initialize the single neuron neural network
    preprocess = Preprocess()
    neural_network = NeuralNetwork()
    
    train_input =   preprocess.get_training_inputs().to_numpy().reshape(-1,11)    
    train_output = preprocess.get_training_outputs().to_numpy().reshape(-1,1)    
    
    test_input = preprocess.get_test_inputs().to_numpy().reshape(-1,11)  
    test_output = preprocess.get_test_outputs().to_numpy().reshape(-1,1)  

        
    train_input = train_input.astype(int)
    train_output= train_output.astype(int)
    
    test_input= test_input.astype(int)
    test_output= test_output.astype(int)
    
    
    neural_network.train(train_input, train_output, 1000)
    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)
    
    
    predicted_outputs = neural_network.think(test_input).astype(int)
    
    predicted_outputs[:,0] = predicted_outputs[:,0] == 1
    print("predicted_outputs: ")
    print(predicted_outputs[:])
    
    true_positive = []   # 1 - 1
    false_positive = []  # 0 - 1
    false_negative = []  # 1 - 0
    true_negative = []   # 0 - 0
    
    count = 0
    
    for i,j in zip(test_output,predicted_outputs):
        if(i == 1 and j == 1):
            true_positive.append(count)
        elif(i == 0 and j == 1):
            false_positive.append(count)
        elif(i == 1 and j == 0):
            false_negative.append(count)
        elif(i == 0 and j == 0):
            true_negative.append(count)
        count += 1
    
    
    
    
    out_arr = np.logical_xor(predicted_outputs, test_output) 
    out_arr = 1 - out_arr
    
    
    true_length = out_arr.sum()
    total_length = out_arr.size
    
    print(true_length)
    print(total_length)
    print(true_length/total_length*100)

      
# =============================================================================
#     training_inputs = preprocess.get_training_inputs()
#     training_outputs = preprocess.get_training_outputs()
# 
#     # Train the neural network
#     neural_network.train(training_inputs, training_outputs, 10000)
# 
#     predicted_outputs = neural_network.think(preprocess.get_training_inputs())
#     print("predicted_outputs: ")
#     print(predicted_outputs)
# 
#     print("expected outputs: ")
#     print(preprocess.get_validate_outputs())
# =============================================================================
