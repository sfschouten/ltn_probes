import torch


class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, hidden_states, dataset):
        assert len(hidden_states) == len(dataset)

        self.hidden_states = hidden_states
        self.sentences = dataset['sentence']
        self.labels = dataset['labels']

    def __getitem__(self, idx):
        return self.hidden_states[idx], self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.hidden_states)


class CustomDatasetTest(torch.utils.data.Dataset):

    def __init__(self, hidden_states, dataset):
        assert len(hidden_states) == len(dataset)

        self.hidden_states_odd = hidden_states[1::2] # gordon is carnivore
        self.hidden_states_even = hidden_states[0::2]  # gordon is GS
        self.sentences_even =[dataset['sentence'][i] for i in range(len(dataset['sentence'])) if i % 2 == 0]
        self.sentences_odd = [dataset['sentence'][i] for i in range(len(dataset['sentence'])) if i % 2 != 0]
        self.labels_even = [dataset['labels'][i] for i in range(len(dataset['labels'])) if i % 2 == 0]
        self.labels_odd = [dataset['labels'][i] for i in range(len(dataset['labels'])) if i % 2 != 0]
        print("dataset loaded")

    def __getitem__(self, idx):
        return self.hidden_states_even[idx],self.hidden_states_odd[idx],  self.sentences_even[idx],self.sentences_odd[idx],self.labels_even[idx],self.labels_odd[idx]

    def __len__(self):
        return len(self.hidden_states_even)
