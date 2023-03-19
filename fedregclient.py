import torch
from torch import nn
import time
from torch.utils.data import DataLoader
from utils import CustomDataset
import copy
from utils import FSGM
class FedRegClientUpdate(object):
    def __init__(self, dataset, batchSize, learning_rate, epochs, idxs, mu, algorithm):
        self.train_loader = DataLoader(CustomDataset(dataset, idxs), batch_size=batchSize, shuffle=True)
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mu = mu
    def train(self, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.5)
        # use the weights of global model for proximal term calculation
        global_model = copy.deepcopy(model)
        # calculate local training time
        start_time = time.time()
        e_loss = []
        for epoch in range(1, self.epochs+1):
            train_loss = 0.0
            model.train()
            for data, labels in self.train_loader:
                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()
                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss + the proximal term
                _, pred = torch.max(output, 1)
                loss = criterion(output, labels.flatten())
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item()*data.size(0)
                # average losses
                train_loss = train_loss/len(self.train_loader.dataset)
                e_loss.append(train_loss)
        total_loss = sum(e_loss)/len(e_loss)
        return model.state_dict(), total_loss, (time.time() - start_time)