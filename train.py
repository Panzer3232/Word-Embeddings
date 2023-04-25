import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import model as m
import data_load as dl
import visualize as v

TextClassifier = m.TextClassifier

# Initialize the model, loss function, and optimizer
model = TextClassifier(dl.X_train.shape[1], 128, len(dl.newsgroups_train.target_names)).to(dl.device)
criterion = nn.CrossEntropyLoss().to(dl.device)
optimizer = optim.Adam(model.parameters())

# In[7]:


train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []

# In[8]:


# Train the model
for epoch in range(dl.EPOCHS):
    total = 0
    correct = 0
    running_loss = 0.0
    for X, y in tqdm(dl.train_loader, desc=f"Epoch {epoch + 1}/{dl.EPOCHS}"):
        optimizer.zero_grad()
        X, y = X.to(dl.device), y.to(dl.device)
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    epoch_loss = running_loss / len(dl.train_data)
    epoch_acc = correct / total
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc)
    print(f"Epoch {epoch + 1}/{dl.EPOCHS} loss: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for X, y in tqdm(dl.test_loader, desc="Testing"):
            X, y = X.to(dl.device), y.to(dl.device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    epoch_loss = running_loss / len(dl.test_data)
    epoch_acc = correct / total
    test_loss_history.append(epoch_loss)
    test_acc_history.append(epoch_acc)
    print(f"Test accuracy: {correct / total}")

v.visualize_results(train_loss_history, train_acc_history, test_loss_history, test_acc_history)