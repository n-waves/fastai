import fire
from fastai.text import *

from sampled_sm import *
import sentencepiece as sp
from sklearn.model_selection import train_test_split
from sputils import *

UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3
UP_ID  = 4

# def get_word_loss(tokens_fraction):
#     def word_loss(preds, targets):
#         return F.cross_entropy(preds, targets) * tokens_fraction
#     return word_loss


def train_lm(dir_path, cuda_id, cl=1, bs=64, backwards=False, lr=3e-4, sampled=True,
             pretrain_id='', sentence_piece_model='sp-100k.model', batch_sets=1, em_sz=400, nh=1150, nl=3, nth=1):
    print(f'dir_path {dir_path}; cuda_id {cuda_id}; cl {cl}; bs {bs}; '
          f'backwards {backwards}; lr {lr}; sampled {sampled}; '
          f'pretrain_id {pretrain_id} batch_sets {batch_sets} em_sz {em_sz} nh {nh} nl {nl}')
    if not hasattr(torch._C, '_cuda_setDevice'):
        print('CUDA not available. Setting device=-1.')
        cuda_id = -1
    torch.cuda.set_device(cuda_id)
    PRE  = 'bwd_' if backwards else 'fwd_'
    IDS = 'ids'
    p = Path(dir_path)
    assert p.exists(), f'Error: {p} does not exist.'
    bptt=70

    opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    with open(p / f'tmp/train_sentences.txt', 'r') as f:
        sentences = f.readlines()[::nth]

    trn_sentences, val_sentences = train_test_split(sentences, test_size=0.025, random_state=12345)
    spp = sp.SentencePieceProcessor()
    spp.Load(str(p / 'tmp' / sentence_piece_model))
    spp.SetEncodeExtraOptions("bos:eos:reverse" if backwards else "bos:eos")
    vs = spp.GetPieceSize()

    val_lm = parallel_encode(spp, val_sentences, shuffle=False, sample=False)
    #trn_lm = lambda:parallel_encode(spp, trn_sentences, train=True)
    trn_lm = parallel_encode(spp, trn_sentences, shuffle=True, sample=False)

    #trn_dl = ReloadableLanguageModelLoader(trn_lm, bs, bptt)
    trn_dl = LanguageModelLoader(trn_lm, bs, bptt)
    val_dl = LanguageModelLoader(val_lm, bs//5 if sampled else bs, bptt, batch_sets=1)
    md = LanguageModelData(p, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)

    tprs = [1] #get_prs(trn_lm, vs)
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*0.5
    learner,crit = get_learner(drops, 15000, sampled, md, em_sz, nh, nl, opt_fn, tprs)
    wd=1e-7
    learner.metrics = [accuracy]

    lrs = np.array([lr/6,lr/3]+[lr]*(nl-1))
    #lrs=lr
    best=f'best_{PRE}{pretrain_id}'
    if Path(learner.get_model_path(best)).exists():
        learner.load(best)
    learner.fit(lrs, 1, wds=wd, use_clr=(32,10), cycle_len=cl, best_save_name=best)
    learner.save(f'{PRE}{pretrain_id}')
    learner.save_encoder(f'{PRE}{pretrain_id}_enc')

if __name__ == '__main__': fire.Fire(train_lm)
