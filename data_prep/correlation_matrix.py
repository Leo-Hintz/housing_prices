import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/Housing.csv")
df["mainroad"] = df["mainroad"].map(lambda x: 1 if x == "yes" else 0)
df["guestroom"] = df["guestroom"].map(lambda x: 1 if x == "yes" else 0)
df["basement"] = df["basement"].map(lambda x: 1 if x == "yes" else 0)
df["hotwaterheating"] = df["hotwaterheating"].map(lambda x: 1 if x == "yes" else 0)
df["airconditioning"] = df["airconditioning"].map(lambda x: 1 if x == "yes" else 0)
df["prefarea"] = df["prefarea"].map(lambda x: 1 if x == "yes" else 0)
df["furnishingstatus"] = df["furnishingstatus"].map(lambda x: 2 if x == "furnished" else 1 if x == "semi-furnished" else 0)

correleation_matrix = df.corr() 

plt.figure(figsize=(10, 10))
sns.heatmap(correleation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
