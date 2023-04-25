# Text-Classifier for 20 Newsgroups Dataset

This project is an implementation of a simple text classifier using feedforward neural network to classify newsgroup posts into 20 different categories.

## Project Pipeline
### Load the dataset
20 Newsgroups dataset is loaded and divided into training and testing sets.
### Text preprocessing
a) Text data is preprocessed using the pipeline that consists of CountVectorizer, TfidfTransformer and sklearn.feature_extraction.text.<br>
b) Text is transformed into matrix of term-document frequencies and IDF normalization is applied.<br>
### Create Data Loaders
Preprocessed data is converted into Pytorch tensors and loaded into 'DataLoader'
### Neural Network Model
a) A simple feedforward neural network is defined with one hidden layer, dropout,and ReLu activations functions.<br>
b) Cross Entropy Loss and Adam optimizer is used.<br>
### Model Training 
a) Model is trained for a specified number of epochs.<br>
b) Loss and accuracy for each epoch is recorded.
### Evaluation of Model
Model is evaluated using test data loader and test accuracy is computed.
### Plotting
Loss and accuracy history for both training and test sets are plotted using 'matplotlib'

## Run the below command to execute the python scripts
<code> python train.py </code>
