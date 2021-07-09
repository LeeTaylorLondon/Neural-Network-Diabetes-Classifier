# Neural-Network-Diabetes-Classifier
Neural network which is trained on patient data to classify whether someone is diabetic. 
The training data contains 8 data points about the patient and is around 600 training samples. 
This project is inspired by the book, Neural Network Projects with Python by James Loy.

## Testing
See file, results.docx, for different model configurations tested. Overall, from testing a range of 
different layers, units, optimizers and losses recorded and unrecorded. I found that each model's 
accuracy always achieved between 69% and 81%. According to the book, this is due to the lack of features
and training examples.

## Installation
* Pip install h5py (built with 3.1.0)
* Pip install tensorflow (built with 2.5.0)
* Pip install numpy (built with 1.19.5)
* Download data from https://www.kaggle.com/uciml/pima-indians-diabetes-database and place in the same dir as main.py

## Usage
Before running any python files please make sure you have completed ALL above installation steps.

Read main.py for the structure of the model. Run main.py to output the training and testing accuracy 
of the model.

## Neural Network Details
Input Dimensions (8)  
Hidden Layer 1 - 32 Units - Activation Relu  
Hidden Layer 2 - 16 Units - Activation Relu  
Output Layer - 1 Unit - Activation Sigmoid  
Optimizer Adam - Loss Binary Crossentropy  

Accuracy after multiple tests  
Training Accuracy: 91.00% - 99.90%  
Testing Accuracy: 69.00% - 81.00%

## Credits
* Author: Lee Taylor
