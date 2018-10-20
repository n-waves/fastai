from torch.utils.data import Dataset
import numpy as np

class SPDataset(Dataset):
  def __init__(self, spp, sentences, labels):
    self.spp = spp
    self.sentences = sentences
    self.labels = labels

  def __len__(self):
    return len(self.sentences)

class SampleEncodeDataset(SPDataset):
  def __init__(self, spp, sentences, labels, alpha=0.1, n=64):
    super().__init__(spp, sentences, labels)
    self.alpha = alpha
    self.n = n
  def __getitem__(self, index):
    return np.array(self.spp.SampleEncodeAsIds(self.sentences[index], self.n, self.alpha)), self.labels[index]

class BestEncodeDataset(SPDataset):
  def __getitem__(self, index):
    return np.array(self.spp.EncodeAsIds(self.sentences[index])), self.labels[index]

