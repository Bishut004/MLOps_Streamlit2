import pandas as pd
from sklearn.model_selection import train_test_split

# 1. LOAD DATASET
df = pd.read_csv("tourism_project/data/tourism.csv")

# 2. DROP CUSTOMER IDENTIFIER COLUMN
df.drop(columns=["CustomerID"], inplace=True)

# 3. SEPARATE FEATURES AND TARGET
X = df.drop(columns=["ProdTaken"])
y = df["ProdTaken"]

# 4. SPLIT DATA INTO TRAIN AND TEST SETS
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5. SAVE TRAIN AND TEST DATASETS
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# 6. DISPLAY DATASET PREPARATION INFORMATION
print("Data preparation completed successfully!")

print("\nDataset dimensions:")
print("Original dataset:", df.shape)
print("Xtrain:", Xtrain.shape)
print("Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape)
print("ytest:", ytest.shape)

print("\nTarget distribution:")
print("Original dataset:")
print(y.value_counts(normalize=True))

print("\nTraining dataset:")
print(ytrain.value_counts(normalize=True))

print("\nTesting dataset:")
print(ytest.value_counts(normalize=True))
