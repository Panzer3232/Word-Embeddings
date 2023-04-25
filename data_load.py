import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 10

# Download and extract the dataset
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# Define the text preprocessing pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer())
])

# Preprocess the training and testing data
X_train = pipeline.fit_transform(newsgroups_train.data).todense()
y_train = newsgroups_train.target
X_test = pipeline.transform(newsgroups_test.data).todense()
y_test = newsgroups_test.target

# Convert the data into PyTorch tensors and create the data loaders
train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train))
test_data = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)