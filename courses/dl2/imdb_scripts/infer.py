import fire
from fastai.text import *

from sampled_sm import *
import sentencepiece as sp


UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3


class LMTextDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __getitem__(self, idx):
        sentence = self.x[idx]
        return sentence[:-1], sentence[1:]

    def __len__(self): return len(self.x)


def get_lm(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token, bidir=False,
           tie_weights=True, qrnn=False):
    rnn_enc = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir, qrnn=qrnn)
    enc = rnn_enc.encoder if tie_weights else None
    return SequentialRNN(rnn_enc, LinearDecoder(n_tok, emb_sz, 0, tie_encoder=enc))


def predict(lm, dl, spp):
    loss = 0.0
    with no_grad_context():
        for (x, y) in dl:
            targets = y.view(-1)
            preds = lm(x)[0]
            not_pads = targets != PAD_ID
            ce = F.cross_entropy(preds[not_pads], targets[not_pads], reduction='sum')
            loss += ce
    return loss


def infer(dir_path, test_set, cuda_id, bs=64, pretrain_id='', sentence_piece_model='sp-100k.model', correct_for_up=True,
          limit=None, em_sz=400, nh=1150, nl=3):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; bs {bs}; '
          f'pretrain_id {pretrain_id} em_sz {em_sz} nh {nh} nl {nl}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)
    PRE  = 'fwd_'
    p = Path(dir_path)
    pretrain_id = pretrain_id if pretrain_id == '' else f'{pretrain_id}_'
    lm_file = f'{PRE}{pretrain_id}enc'
    lm_path = p / 'models' / f'{lm_file}.h5'

    assert p.exists(), f'Error: {p} does not exist.'
    bptt=70


    test_ids = np.load(p / test_set)
    if isinstance(test_ids[0], list):
        test_ids = [np.array(x) for x in test_ids]
    if limit is not None:
        test_ids = test_ids[::limit]

    test_ds = LMTextDataset(test_ids)
    test_samp = SortSampler(test_ids, key=lambda x: len(test_ids[x]))
    test_dl = DataLoader(test_ds, bs, transpose=True, transpose_y=True, num_workers=1, pad_idx=PAD_ID, sampler=test_samp, pre_pad=False)
    md = ModelData(dir_path, None, test_dl)

    spp = sp.SentencePieceProcessor()
    spp.Load(str(p / 'tmp' / sentence_piece_model))
    vs = spp.GetPieceSize()
    if correct_for_up:
        tokens_total = (len(spp.DecodeIds(np.concatenate(test_ids).tolist()).split(' ')) + (
                np.concatenate(test_ids) == EOS_ID).sum() - (np.concatenate(test_ids) == 4).sum())
    else:
        tokens_total = (len(spp.DecodeIds(np.concatenate(test_ids).tolist()).split(' ')) + (np.concatenate(test_ids) == EOS_ID).sum())
    lm = get_lm(bptt, 1000000, vs, em_sz, nh, nl, PAD_ID)
    lm = to_gpu(lm)
    load_model(lm[0], lm_path)
    lm.reset()
    lm.eval()

    loss = predict(lm, test_dl, spp) / float(tokens_total)
    print(f'Cross entropy: {loss}\nPerplexity: {np.exp(loss)}')

if __name__ == '__main__': fire.Fire(infer)
