from torchvision.transforms import transforms

config = {
    "algorithm": "fedavg",
    "data_flag": "dermamnist",
    "download": True,
    "data_path": "./data",
    "data_transform": transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[.5], std=[.5])
                                        ]),
    "iid": True,
    "num_clients": 10,
    "num_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.1,
    "mu": 0.1,
    "num_clients": 10,
    "frac_clients": 0.2, # fraction of clients
    "drop_percentage": 10, # example: 10 for 10%
    "adaptive": False
}