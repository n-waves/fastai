import fire
from fastai.text import *
from fastai.lm_rnn import *
from sklearn.model_selection import train_test_split
import sentencepiece as sp
from spdatasets import *
from sklearn.metrics import f1_score, precision_score, recall_score

def f1s(preds, targs):
    preds = torch.max(preds, dim=1)[1].cpu().numpy()
    targs = targs.cpu().numpy()
    pa = precision_score(targs, preds, average='binary')
    ra = recall_score(targs, preds, average='binary')
    if pa == 0.0 or ra == 0.0:
      return 0.0
    return 2/(1/pa + 1/ra)

def freeze_all_but(learner, n):
    c=learner.get_layer_groups()
    for l in c: set_trainable(l, False)
    set_trainable(c[n], True)

# FOCAL
def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).cuda().fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).cuda().fill_(1))
        mask = Variable(mask)

    return mask.scatter_(1, index, ones)

def focal_loss(input, target):
    eps, gamma = 1e-7, 2.5
    y = one_hot(target, input.size(-1))
    logit = F.softmax(input, dim=-1)
    logit = logit.clamp(eps, 1. - eps)

    ll = logit.log()

    loss = -y
    loss = loss * ll # cross entropy
    loss = loss * (1 - logit) ** gamma # focal loss

    return loss.mean()
# FOCAL

# F1 loss

def f1_loss(input, target):
    eps = 1e-7
    logit = F.softmax(input, dim=-1) # [bs, 2]
    tp = logit[target == 1, 1].sum()
    tn = logit[target == 0, 0].sum()
    bs = target.shape[0]
    return 1.0 - 2.0*tp / (tp - tn + bs +eps)


class F1_RNN_Learner(RNN_Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return f1_loss

# F! loss
def weighted_cross_entropy(*args,**kwargs):
  return F.cross_entropy(*args, **kwargs, weight=torch.cuda.FloatTensor([1, 1]))

class Weighted_RNN_Learner(RNN_Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return weighted_cross_entropy

def train_clas(dir_path, cuda_id, lm_id='', clas_id=None, bs=64, cl=1, backwards=False, startat=0, unfreeze=True,
               lr=0.01, dropmult=1.0, bpe=False, use_clr=True,
               use_regular_schedule=False, use_discriminative=True, last=False, chain_thaw=False,
               from_scratch=False, train_file_id='', nl=3, sentence_piece_model=None, sp_alpha=0.1, sp_n=64, wd=1e-6, clip=25., vs=25000):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; lm_id {lm_id}; clas_id {clas_id}; bs {bs}; cl {cl}; backwards {backwards}; '
        f'dropmult {dropmult} unfreeze {unfreeze} startat {startat}; bpe {bpe}; use_clr {use_clr};'
        f'use_regular_schedule {use_regular_schedule}; use_discriminative {use_discriminative}; last {last};'
        f'chain_thaw {chain_thaw}; from_scratch {from_scratch}; train_file_id {train_file_id} sp_alpha {sp_alpha} sp_n {sp_n}'
        f'clip {clip} vs {vs}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)

    PRE = 'bwd_' if backwards else 'fwd_'
    PRE = 'bpe_' + PRE if bpe else PRE
    IDS = 'bpe' if bpe else 'ids'
    train_file_id = train_file_id if train_file_id == '' else f'_{train_file_id}'
    dir_path = Path(dir_path)
    lm_id = lm_id if lm_id == '' else f'{lm_id}_'
    clas_id = lm_id if clas_id is None else clas_id
    clas_id = clas_id if clas_id == '' else f'{clas_id}_'
    intermediate_clas_file = f'{PRE}{clas_id}clas_0'
    final_clas_file = f'{PRE}{clas_id}clas_1'
    best_clas_file = f'{PRE}{clas_id}best_clas_1'
    lm_file = f'{PRE}{lm_id}lm_enc'
    lm_path = dir_path / 'models' / f'{lm_file}.h5'
    assert lm_path.exists(), f'Error: {lm_path} does not exist.'

    bptt,em_sz,nh = 70,400,1150
    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

#######
    if backwards:
        trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}{train_file_id}_bwd.npy')
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}_bwd.npy')
    else:
        trn_sent = np.load(dir_path / 'tmp' / f'trn_{IDS}{train_file_id}.npy')
        val_sent = np.load(dir_path / 'tmp' / f'val_{IDS}.npy')

    trn_lbls = np.load(dir_path / 'tmp' / f'lbl_trn{train_file_id}.npy')
    val_lbls = np.load(dir_path / 'tmp' / f'lbl_val.npy')

####
#    with open(dir_path / 'tmp' / f'train.txt', 'r') as f:
#        rows = [line.split('\t') for line in f.readlines()]
#    sentences = [row[0] for row in rows]
#    labels = np.array([[int(row[1] != 'OTHER')] for row in rows])
#
#    trn_sent, val_sent, trn_lbls, val_lbls = train_test_split(sentences, labels, test_size=0.1, random_state=12345)
#
#    spp = sp.SentencePieceProcessor()
#    spp.Load(str(dir_path / 'tmp' / sentence_piece_model))
#    spp.SetEncodeExtraOptions("bos:eos:reverse" if backwards else "bos:eos")
#    vs = spp.GetPieceSize()
####
    assert trn_lbls.shape[1] == 1 and val_lbls.shape[1] == 1, 'This classifier uses cross entropy loss and only support single label samples'
    trn_lbls = trn_lbls.flatten()
    val_lbls = val_lbls.flatten()
    trn_lbls -= trn_lbls.min()
    val_lbls -= val_lbls.min()
    c=int(trn_lbls.max())+1

#    vs = 25000
#    if bpe: vs=30002
#    else:
#        itos = pickle.load(open(dir_path / 'tmp' / 'itos.pkl', 'rb'))
#        vs = len(itos)

    #trn_ds = SampleEncodeDataset(spp, trn_sent, trn_lbls, sp_alpha, sp_n)
    trn_ds = TextDataset(trn_sent, trn_lbls)
    val_ds = TextDataset(val_sent, val_lbls)
    #trn_ds = BestEncodeDataset(spp, trn_sent, trn_lbls)
    #val_ds = BestEncodeDataset(spp, val_sent, val_lbls)
    trn_samp = SortishSampler(trn_sent, key=lambda x: len(trn_sent[x]), bs=bs//2)
    val_samp = SortSampler(val_sent, key=lambda x: len(val_sent[x]))
    trn_dl = DataLoader(trn_ds, bs//2, transpose=True, num_workers=1, pad_idx=1, sampler=trn_samp)
    val_dl = DataLoader(val_ds, bs, transpose=True, num_workers=1, pad_idx=1, sampler=val_samp)
    md = ModelData(dir_path, trn_dl, val_dl)

    dps = np.array([0.4,0.5,0.05,0.3,0.4])*dropmult
    #dps = np.array([0.5, 0.4, 0.04, 0.3, 0.6])*dropmult
    #dps = np.array([0.65,0.48,0.039,0.335,0.34])*dropmult
    #dps = np.array([0.6,0.5,0.04,0.3,0.4])*dropmult

    m = get_rnn_classifier(bptt, 20*70, c, vs, emb_sz=em_sz, n_hid=nh, n_layers=nl, pad_token=1,
              layers=[em_sz*3, 50, c], drops=[dps[4], 0.1],
              dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

    #learn = Weighted_RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn = F1_RNN_Learner(md, TextModel(to_gpu(m)), opt_fn=opt_fn)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip=clip
    learn.metrics = [f1s, accuracy]

    lrm = 2.6
    if use_discriminative:
        lrs = np.array([lr/(lrm**4), lr/(lrm**3), lr/(lrm**2), lr/lrm] + [lr] * (nl-2))
    else:
        lrs = lr
    #wd = 1e-6
    if not from_scratch:
        learn.load_encoder(lm_file)
    else:
        print('Training classifier from scratch. LM encoder is not loaded.')
        use_regular_schedule = True

    if (startat<1) and not last and not chain_thaw and not from_scratch:
        learn.freeze_to(-1)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8,3))
        learn.freeze_to(-2)
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8, 3))
        learn.save(intermediate_clas_file)
    elif startat==1:
        learn.load(intermediate_clas_file)

    if chain_thaw:
        lrs = np.array([0.0001, 0.0001, 0.0001, 0.0001] + [0.001] * (nl-2))
        print('Using chain-thaw. Unfreezing all layers one at a time...')
        n_layers = len(learn.get_layer_groups())
        print('# of layers:', n_layers)
        # fine-tune last layer
        learn.freeze_to(-1)
        print('Fine-tuning last layer...')
        learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                  use_clr=None if use_regular_schedule or not use_clr else (8,3))
        n = 0
        # fine-tune all layers up to the second-last one
        while n < n_layers-1:
            print('Fine-tuning layer #%d.' % n)
            freeze_all_but(learn, n)
            learn.fit(lrs, 1, wds=wd, cycle_len=None if use_regular_schedule else 1,
                      use_clr=None if use_regular_schedule or not use_clr else (8,3))
            n += 1

    if unfreeze:
        learn.unfreeze()
    else:
        learn.freeze_to(-3)

    if last:
        print('Fine-tuning only the last layer...')
        learn.freeze_to(-1)

    if use_regular_schedule:
        print('Using regular schedule. Setting use_clr=None, n_cycles=cl, cycle_len=None.')
        use_clr = None
        n_cycles = cl
        cl = None
    else:
        n_cycles = 1
    learn.fit(lrs, n_cycles, wds=wd, cycle_len=cl, use_clr=(8,8) if use_clr else None, best_save_name=best_clas_file)
    print('Plotting lrs...')
    learn.sched.plot_lr()
    learn.save(final_clas_file)

if __name__ == '__main__': fire.Fire(train_clas)

