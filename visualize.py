import matplotlib.pyplot as plt

def visualize_results(train_loss_history, train_acc_history, test_loss_history, test_acc_history):
    # Plot the loss and accuracy for each epoch
    plt.plot(train_loss_history, label="Training loss")
    plt.plot(test_loss_history, label="Test loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    plt.plot(train_acc_history, label="Training accuracy")
    plt.plot(test_acc_history, label="Test accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()