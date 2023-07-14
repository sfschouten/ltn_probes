import torch


class CustomDataset(torch.utils.data.Dataset):

  def __init__(self,dataset,labels):

    # Your code

    self.instances = dataset
    self.labels= labels["labels"]

  def __getitem__(self, idx):
    return self.instances[idx],self.labels[idx] # In case you stored your data on a list called instances

  def __len__(self):
    return len(self.instances)