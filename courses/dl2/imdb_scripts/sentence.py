import fire
from fastai.text import *

from sampled_sm import *
import sentencepiece as sp
import readline

UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3
UP_ID  = 4

def id2piece(spp, i):
  if i == 0:
      return '<unk>'
  return spp.IdToPiece(i)

def get_prs(m, spp, z):
    s = spp.EncodeAsIds(z) + [EOS_ID]
    t = LongTensor(s).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    m[0].bs=1
    m.eval()
    m.reset()
    res, *_ = m(t)
    res = F.log_softmax(res, dim=1)
    pr = 0.0
    for i, tok in enumerate(s[1:]):
        pr += res[i, tok].item()
    print(f'{pr:0.4f}: {z}')

def inter_sample_model(m, spp, s, l=50):
    s = spp.EncodeAsIds(s)
    print(' '.join(spp.IdToPiece(x) for x in s), end='')
    t = LongTensor(s).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    m[0].bs=1
    m.eval()
    m.reset()
    print('...') #, end='')
    pr = 0.0
    for i in range(l):
        res,*_ = m(t)

        r=F.log_softmax(res[-1], dim=0).topk(5)
        for j in range(5):
          print(f'{j+1}: {id2piece(spp, r[1][j].item())} {r[0][j].item():0.4f}')
        a = int(input())
        if a == 0:
          break
        n=res[-1].topk(2)[1]
        n = r[1][a-1] # n[1] if n.data[0]==0 else n[0]
        pr += r[0][a-1]
        n_id = int(n.item())
        word = id2piece(spp, n_id)
        print(word) #, end=' ')
        if n_id == EOS_ID: break
        t = torch.cat((t, n.unsqueeze(0).unsqueeze(0)))
    print(f'{pr:0.4f}')

def sample_model(m, spp, s, l=50):
    s = spp.EncodeAsIds(s)
    print(' '.join(spp.IdToPiece(x) for x in s), end='')
    t = LongTensor(s).view(-1,1).cuda()
    t = Variable(t,volatile=False)
    m[0].bs=1
    m.eval()
    m.reset()
    print('...', end='')

    for i in range(l):
        res,*_ = m(t)
        n=res[-1].topk(2)[1]
        n = n[0] # n[1] if n.data[0]==0 else n[0]
        n_id = int(n.item())
        word = id2piece(spp, n_id)
        print(word, end=' ')
        if n_id == EOS_ID: break
        t = torch.cat((t, n.unsqueeze(0).unsqueeze(0)))
    print()

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


def predict(lm, dl, use_tqdm=True):
    loss = 0.0
    with no_grad_context():
        for (x, y) in tqdm(dl, disable=(not use_tqdm)):
            targets = y.view(-1)
            preds = lm(x)[0]
            not_pads = targets != PAD_ID
            ce = F.cross_entropy(preds[not_pads], targets[not_pads], reduction='sum')
            loss += ce
    return loss

def calc_statistics(spp_model_path, correct_for_up):
    spp = sp.SentencePieceProcessor()
    spp.Load(str(spp_model_path))
    spp.SetEncodeExtraOptions("bos")
    vs = spp.GetPieceSize()
    stats = {"vs": vs}
    return stats, spp

def infer(dir_path, cuda_id, bs=64, pretrain_id='', sentence_piece_model='sp-100k.model', correct_for_up=True,
          limit=None, em_sz=400, nh=1150, nl=3, use_tqdm=True):
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    pretrain_id = pretrain_id if pretrain_id == '' else f'{pretrain_id}_'
    p = Path(dir_path)
    print(f'{os.getpid()}: dir_path {dir_path}; cuda_id {cuda_id}; bs {bs}; limit: {limit}; '
          f'pretrain_id {pretrain_id} em_sz {em_sz} nh {nh} nl {nl}')
    def prepare():
        PRE  = 'fwd_'

        lm_file = f'{PRE}{pretrain_id}enc'
        lm_path = p / 'models' / f'{lm_file}.h5'

        assert p.exists(), f'Error: {p} does not exist.'
        bptt=5

        stats, spp = calc_statistics(p / 'tmp' / sentence_piece_model, correct_for_up)

        lm = get_lm(bptt, 1000000, stats['vs'], em_sz, nh, nl, PAD_ID)
        lm = to_gpu(lm)
        load_model(lm[0], lm_path)
        lm.reset()
        lm.eval()
        return lm, stats, spp

    lm, stats, spp  = prepare()
    print(stats)
    while True:
        s = input("Enter a sentence: ")
        if s.strip() == '': break
        if '$x' in s:
            v = input("Enter values: ")
            v = [x.strip() for x in v.split(',')]
            for x in v:
                get_prs(lm, spp, s.replace('$x', x))
        elif s.endswith('>'):
            inter_sample_model(lm, spp, s[:-1])
        else:
            sample_model(lm, spp, s)



if __name__ == '__main__': fire.Fire(infer)
