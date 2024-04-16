import network as network
import data as data
import torch
import matplotlib.pyplot as plt
net = network.Net()
net.load_state_dict(torch.load("nn.pth"))
dataset = data.HousingDataset("data/Housing.csv")


y_pred = ((net.forward(dataset.features.unsqueeze(1)) * (dataset.max_price - dataset.min_price)) + dataset.min_price).detach().numpy()
plt.figure(figsize=(10, 10))


x = x = dataset.features[:, 0].detach().numpy()
y = dataset.prices.detach().numpy()
plt.scatter(x, y, label="Actual Price")
plt.scatter(x, y_pred, label="Predicted Price")
plt.title("Training Set Prediction")
plt.legend()
plt.grid(True)
plt.show()
