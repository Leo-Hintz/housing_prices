# Trains a neural network on the Housing dataset and saves the model to model.pth
import torch
import network
import data
import matplotlib.pyplot as plt
net = network.Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
net.to(device)
dataset = data.HousingDataset("data/Housing.csv")
losses, weights = net.train(dataset, device)
torch.save(net.state_dict(), "nn.pth")

plt.figure(figsize=(10, 10))

plt.subplot(2, 1, 1)
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(weights)
plt.xlabel("Epochs")
plt.ylabel("Mean Weight")
plt.title("Mean Weight")
plt.grid(True)

plt.show()
