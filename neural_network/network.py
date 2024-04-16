import torch
import torch.nn as nn

BATCH_SIZE = 90
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6, 800)
        self.layer2 = nn.ReLU()
        self.layer3 = nn.Linear(800, 800)
        self.layer4 = nn.ReLU()
        self.layer5 = nn.Linear(800, 800)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Linear(800, 800)
        self.layer8 = nn.Sigmoid()
        self.layer9 = nn.Linear(800, 1)
        
        # set weights
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.zeros_(self.layer3.bias)
        nn.init.xavier_uniform_(self.layer5.weight)
        nn.init.zeros_(self.layer5.bias)
        nn.init.xavier_uniform_(self.layer7.weight)
        nn.init.zeros_(self.layer7.bias)
        nn.init.xavier_uniform_(self.layer9.weight)
        nn.init.zeros_(self.layer9.bias)
        
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        return x
    
    def train(self, x, device):
        losses = []
        weights = []
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=.9)
        criterion = nn.MSELoss()
        trainloader = torch.utils.data.DataLoader(x, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        for epoch in range(25):
            total_loss = 0
            for _, data in enumerate(trainloader, 0):
                input, y = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                y_hat = self.forward(input.unsqueeze(1))
                loss = criterion(y_hat, y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                
                # get average weight
                loss_cpu = loss.cpu().item()
                total_loss += loss_cpu / len(trainloader)
            mean_weight = sum(p.data.cpu().mean() for p in self.parameters())
            weights.append(mean_weight)
            print(f"Epoch: %d, Loss: {total_loss:.2e}" % epoch)
            print("Mean Weight: %f" % mean_weight)
            losses.append(total_loss)
        return losses, weights
