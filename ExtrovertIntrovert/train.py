import torch
import torch.nn as nn
import argparse

from sklearn.model_selection import train_test_split

from simpleMLP import simpleMLP
from loadAndCleanUpDataset import loadAndCleanUpDataset, turnDatasetIntoTorchTensor
from extroIntroDataset import extroIntroDataset

def train(dataset, randState = 42, test_size = 0.3):
    #Get a device to train the model on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load a dataframe and tensor that will represent the dataset.
    df = loadAndCleanUpDataset(fn = dataset)
    dataTensor = turnDatasetIntoTorchTensor(df)

    #Split into inputs and outputs
    personalityFeatures = dataTensor[:, 0:7]
    personalityType = dataTensor[:, 7]
    del df, dataTensor #not needed anymore

    #Train Test Split. Can be manually set for repeatability and easier result analysis / debugging
    X_train, _, y_train, _ = train_test_split(personalityFeatures, personalityType, test_size=test_size, random_state=randState)
    del personalityFeatures, personalityType, _

    #Put the dataset into the torch dataset for use in a dataloader
    trainDataset = extroIntroDataset(X_train, y_train)
    del X_train, y_train
    
    #Get the model, criterion
    model = simpleMLP(7)
    criterion = nn.BCEWithLogitsLoss() #Sigmoid with binary cross entropy so that sigmoid does not need to be done in the model 
    
    #Place Train Loop Here

    return

def main(args = None):
    parser = argparse.ArgumentParser(description = "Training loop for the kaggle dataset: https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data")
    parser.add_argument('--dataset', type = str, default = "personality_dataset.csv") 
    args = parser.parse_args()
    train(dataset = args.dataset)
    return

if __name__ == "__main__":
    main()

