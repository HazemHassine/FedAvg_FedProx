import medmnist
from medmnist import INFO
import importlib
import argparse
from models import ResNet18
from server import training
import numpy as np

from utils import iid_partition, non_iid_partition


def main():
    argumentparser = argparse.ArgumentParser()
    argumentparser.add_argument("-c", "--config", help="path to the config file")
    arguments = argumentparser.parse_args()
    config = importlib.import_module(arguments.config.replace("/","."),package=None)
    config = config.config
    
    algorithm = config["algorithm"]
    data_flag = config["data_flag"]
    iid = config["iid"]

    # download data and make 
    info = INFO[data_flag]
    task = info['task']
    DataClass = getattr(medmnist, info['python_class'])
    download = config["download"]
    data_path = config["data_path"]
    transforms = config["data_transform"]
    # load the data
    train_dataset = DataClass(root=data_path, split='train', transform=transforms, download=download)
    test_dataset = DataClass(root=data_path, split='test', transform=transforms, download=download)
    num_channels = train_dataset[0][0].shape[0]
    num_classes = len(np.unique(train_dataset.labels))
    model = ResNet18(in_channels=num_channels, num_classes=num_classes)
    batch_size = config["batch_size"]
    lr = config["learning_rate"]
    drop_percentage = config["drop_percentage"]
    rounds = config["num_epochs"]
    frac_clients = config["frac_clients"]
    num_clients = config["num_clients"]
    mu = config["mu"]
    adaptive = config["adaptive"]
    epochs = config["num_epochs"]
    print("##################### DATASET #####################")
    print(train_dataset)
    print("##################### TRAINING #####################")
    print("training using: " + algorithm.upper())
    if iid:
        data_dict = iid_partition(train_dataset, num_clients)
    else:
        data_dict = non_iid_partition(train_dataset, num_clients)
        
    training(model, rounds, batch_size, lr, train_dataset, data_dict, test_dataset, frac_clients, num_clients, epochs, mu,adaptive ,drop_percentage, 99.9, algorithm=algorithm.lower())
    

if __name__=="__main__":
    main()