import pandas as pd

def load_data(file):
    data = pd.read_csv(file)
    data["mainroad"] = data["mainroad"].map(lambda x: 1 if x == "yes" else 0)
    data["guestroom"] = data["guestroom"].map(lambda x: 1 if x == "yes" else 0)
    data["basement"] = data["basement"].map(lambda x: 1 if x == "yes" else 0)
    data["hotwaterheating"] = data["hotwaterheating"].map(lambda x: 1 if x == "yes" else 0)
    data["airconditioning"] = data["airconditioning"].map(lambda x: 1 if x == "yes" else 0)
    data["prefarea"] = data["prefarea"].map(lambda x: 1 if x == "yes" else 0)
    data["furnishingstatus"] = data["furnishingstatus"].map(lambda x: 2 if x == "furnished" else 1 if x == "semi-furnished" else 0)
    return data.drop("price", axis=1), data["price"]