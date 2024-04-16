import torch
from torch.utils.data import Dataset
import pandas as pd

SKIPPED_ROWS = 10
MAX_ROWS = 500
# expected input vector: [area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishing status]

class HousingDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = pd.read_csv(data_path, skiprows=range(1, SKIPPED_ROWS + 1), nrows=MAX_ROWS)
        self.transform = transform
        self.features = []
        self.prices = []
        for i in range(0, self.data.shape[0]):
            price_tensor = torch.tensor(self.data.iloc[i, 0]).float()
        
            input = (self.data.iloc[i, 1], self.data.iloc[i, 2], self.data.iloc[i, 3], self.data.iloc[i, 4], 1.0 if self.data.iloc[i, 9] == "yes" else 0.0, 1.0 if self.data.iloc[i, 10] == "yes" else 0.0)
            feature_tensor = torch.tensor(input).float()
            #features = features.map(lambda x: 1 if x == "yes" else 0 if x == "no" else x)
        
            #feature_tensor = torch.tensor((features["area"], features["bedrooms"], features["bathrooms"], features["stories"], features["mainroad"], features["guestroom"], features["basement"], features["hotwaterheating"], features["airconditioning"], features["parking"], features["prefarea"]))
        
            #if features["furnishingstatus"] == "furnished":
            #    feature_tensor = torch.cat((feature_tensor, torch.tensor([1,0,0]).float()))
            #elif features["furnishingstatus"] == "semi-furnished":
            #    feature_tensor = torch.cat((feature_tensor, torch.tensor([0,1,0]).float()))
            #else:
            #    feature_tensor = torch.cat((feature_tensor, torch.tensor([0,0,1]).float()))
            self.features.append(feature_tensor)
            self.prices.append(price_tensor)
        
        

        self.features = torch.stack(self.features)
        self.prices = torch.stack(self.prices)

        # Get values for normalization (Source: https://stats.stackexchange.com/questions/70801/how-to-normalize-data-to-0-1-range)
        self.min_feature = torch.min(self.features)
        self.max_feature = torch.max(self.features)

        self.min_price = torch.min(self.prices)
        self.max_price = torch.max(self.prices)

    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        return (self.features[idx] - self.min_feature) / (self.max_feature - self.min_feature), (self.prices[idx] - self.min_price) / (self.max_price - self.min_price)