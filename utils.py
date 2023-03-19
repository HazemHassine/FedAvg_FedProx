from torch.utils.data import Dataset
import numpy as np


import torch

def FSGM(model, inp, label, iters=5, eta=0.1):
    inp.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()
    minv, maxv = float(inp.min().detach().cpu().numpy()), float(inp.max().detach().cpu().numpy())
    # print(inp.shape)
    # print(label.shape)
    print(eta)
    for _ in range(iters):
        loss = criterion(model.forward(inp), label.flatten().long()).mean()
        dp = torch.sign(torch.autograd.grad(loss, inp)[0])
        inp.data.add_(eta*dp.detach()).clamp(minv, maxv)
    return inp

def non_iid_partition(dataset, num_clients):
    """
    non I.I.D parititioning of data over clients
    Sort the data by the digit label
    Divide the data into N shards of size S
    Each of the clients will get X shards
    params:
      - dataset (torch.utils.Dataset): Dataset containing the pathMNIST Images
      - num_clients (int): Number of Clients to split the data between
      - total_shards (int): Number of shards to partition the data in
      - shards_size (int): Size of each shard 
      - num_shards_per_client (int): Number of shards of size shards_size that each client receives
    returns:
      - Dictionary of image indexes for each client
    """
    shards_size = 9
    total_shards = len(dataset)// shards_size
    num_shards_per_client = total_shards // num_clients
    shard_idxs = [i for i in range(total_shards)]
    client_dict = {i: np.array([], dtype='int64') for i in range(num_clients)}
    idxs = np.arange(len(dataset))
    # get labels as a numpy array
    data_labels = np.array([target.numpy().flatten() for _, target in dataset]).flatten()
    # sort the labels
    label_idxs = np.vstack((idxs, data_labels))
    label_idxs = label_idxs[:, label_idxs[1,:].argsort()]
    idxs = label_idxs[0,:]
    # divide the data into total_shards of size shards_size
    # assign num_shards_per_client to each client
    for i in range(num_clients):
        rand_set = set(np.random.choice(shard_idxs, num_shards_per_client, replace=False))
        shard_idxs = list(set(shard_idxs) - rand_set)
        for rand in rand_set:
            client_dict[i] = np.concatenate((client_dict[i], idxs[rand*shards_size:(rand+1)*shards_size]), axis=0)
    return client_dict # client dict has [idx: list(datapoint indices)


def iid_partition(dataset, clients):
  """
  I.I.D paritioning of data over clients
  Shuffle the data
  Split it between clients
  
  params:
    - dataset (torch.utils.Dataset): Dataset containing the PathMNIST Images 
    - clients (int): Number of Clients to split the data between
  returns:
    - Dictionary of image indexes for each client
  """

  num_items_per_client = int(len(dataset)/clients)
  client_dict = {}
  image_idxs = [i for i in range(len(dataset))]

  for i in range(clients):
    client_dict[i] = set(np.random.choice(image_idxs, num_items_per_client, replace=False))
    image_idxs = list(set(image_idxs) - client_dict[i])

  return client_dict

def GenerateLocalEpochs(percentage, size, max_epochs):
  ''' Method generates list of epochs for selected clients
  to replicate system heteroggeneity

  Params:
    percentage: percentage of clients to have fewer than E epochs
    size:       total size of the list
    max_epochs: maximum value for local epochs
  
  Returns:
    List of size epochs for each Client Update

  '''
  # if percentage is 0 then each client runs for E epochs
  if percentage == 0:
      return np.array([max_epochs]*size)
  else:
    # get the number of clients to have fewer than E epochs
    heterogenous_size = int((percentage/100) * size)

    # generate random uniform epochs of heterogenous size between 1 and E
    epoch_list = np.random.randint(1, max_epochs, heterogenous_size)

    # the rest of the clients will have E epochs
    remaining_size = size - heterogenous_size
    rem_list = [max_epochs]*remaining_size

    epoch_list = np.append(epoch_list, rem_list, axis=0)
    
    # shuffle the list and return
    np.random.shuffle(epoch_list)

    return epoch_list

class CustomDataset(Dataset):
  def __init__(self, dataset, idxs):
      self.dataset = dataset
      self.idxs = list(idxs)

  def __len__(self):
      return len(self.idxs)

  def __getitem__(self, item):
      image, label = self.dataset[self.idxs[item]]
      return image, label