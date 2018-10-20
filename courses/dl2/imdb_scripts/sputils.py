import torch
from torch.utils.data import Dataset
import numpy as np

class ReloadableLanguageModelLoader():
    def __init__(self, nums_closure, bs, bptt):
        self.bs,self.bptt = bs,bptt
        self.n = None
        self.nums_closure = nums_closure
        #self.data = self.batchify(nums_closure())
        self.i,self.iter = 0,0
        #self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        while self.i < self.n-1 and self.iter<len(self):
            if self.i == 0:
                seq_len = self.bptt + 5 * 5
            else:
                bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
                seq_len = max(5, int(np.random.normal(bptt, 5)))
            res = self.get_batch(self.i, seq_len)
            self.i += seq_len
            self.iter += 1
            yield res
        self.n = None

    def __len__(self):
        if self.n is None:
            self.data = self.batchify(self.nums_closure())
            self.n = len(self.data)
        return self.n // self.bptt - 1

    def batchify(self, data):
        nb = data.shape[0] // self.bs
        data = np.array(data[:nb*self.bs])
        data = data.reshape(self.bs, -1).T
        return T(data)  # .pin_memory()

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)

def parallel_encode(spp, sentences, sample=False,shuffle=False):
  class SPDataset(Dataset):
    def __init__(self, spp, sentences):
      self.spp = spp
      self.sentences = sentences
    def __len__(self):
      return len(self.sentences)

  class SampleEncodeDataset(SPDataset):
    def __init__(self, spp, sentences, alpha=0.1, n=64):
      super().__init__(spp, sentences)
      self.alpha = alpha
      self.n = n
    def __getitem__(self, index):
      return self.spp.SampleEncodeAsIds(self.sentences[index], self.n, self.alpha)
 
  class BestEncodeDataset(SPDataset):
    def __getitem__(self, index):
      return self.spp.EncodeAsIds(self.sentences[index])

  def collate(samples):
    return np.concatenate(samples)

  ds = SampleEncodeDataset(spp, sentences) if sample else BestEncodeDataset(spp, sentences)
  dl = torch.utils.data.DataLoader(ds, batch_size=1000, shuffle=shuffle, num_workers=8, collate_fn=collate)
  return np.concatenate([b for b in dl])

