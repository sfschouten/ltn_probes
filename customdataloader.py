import torch


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, hidden_states, dataset):
        self.hidden_states = hidden_states
        self.sentences = dataset['sentence']
        self.labels = dataset['labels']

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.hidden_states)
