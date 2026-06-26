"""
Lung Cancer Subtyping & Grading — self-contained training + inference pipeline.
Runs locally (CPU smoke test) and on Kaggle T4 / A10G. No local imports (becomes solution.ipynb).

Pipeline: ImageNet-pretrained backbone (fine-tuned) + magnification covariate
  -> StratifiedGroupKFold(patient) OOF -> temperature calibration
  -> Bayes-optimal belief decode + expected-harm referral (exact-grader-tuned)
  -> fold-ensemble test prediction -> ./working/submission.csv

Env knobs: SMOKE=1 (tiny CPU run), BACKBONE, EPOCHS, BS, N_SPLITS, SEEDS, FRAC_GRID.
"""
import os, json, math, random, time, warnings
from pathlib import Path
from math import ceil
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
from PIL import Image

# ----------------------------- config -----------------------------
SEED       = 42
BACKBONE   = os.environ.get("BACKBONE", "convnext_tiny.fb_in22k_ft_in1k")
IMG        = int(os.environ.get("IMG", "224"))
N_SPLITS   = int(os.environ.get("N_SPLITS", "5"))
SEEDS      = [int(s) for s in os.environ.get("SEEDS", "42").split(",")]
EPOCHS     = int(os.environ.get("EPOCHS", "28"))
BS         = int(os.environ.get("BS", "16"))
LR         = float(os.environ.get("LR", "2e-4"))
WD         = float(os.environ.get("WD", "0.05"))
LS         = float(os.environ.get("LS", "0.05"))     # label smoothing
EMA_DECAY  = float(os.environ.get("EMA", "0.999"))
LAMBDA_BIN = float(os.environ.get("LAMBDA_BIN", "0.0"))  # >0 enables aux benign/malignant head
BB_LR_MULT = float(os.environ.get("BB_LR_MULT", "0.3"))  # backbone LR = LR*mult (gentler than head)
FREEZE_EP  = int(os.environ.get("FREEZE_EP", "3"))       # epochs with frozen backbone (head warmup)
TTA        = os.environ.get("TTA", "1") == "1"
SMOKE      = os.environ.get("SMOKE", "0") == "1"
FRAC_GRID  = [float(x) for x in os.environ.get("FRAC_GRID", "0,0.05,0.1,0.15,0.2,0.25,0.3").split(",")]
WORK       = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./working")
WORK.mkdir(exist_ok=True, parents=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------------- metric / decode (verified) -----------------------------
CLASSES = ["nor","aca_bd","aca_md","aca_pd","scc_bd","scc_md","scc_pd"]
CIDX = {c:i for i,c in enumerate(CLASSES)}
MALIG = np.array([0,1,1,1,1,1,1], dtype=bool)
GAMMA_MISS, GAMMA_FA, S_REF = 1.0, 0.5, 0.40
REF_BUDGET, REF_THR = 0.20, 0.5
HARM_FLOOR, HARM_CEIL, SCORE_FLOOR = -2.0, 1.0, 0.02
def _sub(c): return c.split("_")[0]
def _gr(c): return {"bd":1,"md":2,"pd":3}.get(c.split("_")[1],0) if "_" in c else 0
def _aff(t,k):
    if k==t: return 1.0
    if t=="nor" or k=="nor": return -1.0
    if _sub(t)!=_sub(k): return 0.0
    return {1:0.5,2:0.2}[abs(_gr(t)-_gr(k))]
A = np.array([[_aff(t,k) for k in CLASSES] for t in CLASSES])

def harm_score(p, t):
    p=np.asarray(p,float); hr=float(p@A[t])
    pen = GAMMA_MISS*p[0]**2 if t!=0 else GAMMA_FA*(p[MALIG].sum())**2
    return max(HARM_FLOOR, min(HARM_CEIL, hr-pen))

def exact_grade(P, ref, gt):
    P=np.asarray(P,float); ref=np.asarray(ref,float); gt=np.asarray(gt,int); N=len(gt)
    K=int(REF_BUDGET*N); cap=max(1,ceil(K/7))
    elig=sorted([i for i in range(N) if ref[i]>REF_THR], key=lambda i:(-ref[i],i))
    referred=set(); cc=[0]*7
    for i in elig:
        if len(referred)>=K: break
        if cc[gt[i]]<cap: referred.add(i); cc[gt[i]]+=1
    by={c:[] for c in range(7)}
    for i in range(N):
        by[gt[i]].append(S_REF if i in referred else harm_score(P[i],gt[i]))
    macro=float(np.mean([np.mean(v) for v in by.values() if v]))
    return max(SCORE_FLOOR, min(1.0, SCORE_FLOOR+(macro-HARM_FLOOR)/(HARM_CEIL-HARM_FLOOR)*(1.0-SCORE_FLOOR)))

def decode_belief(q, w=None):
    q=np.asarray(q,float); qw=q*(np.ones(7) if w is None else w); qw=qw/qw.sum()
    a=A.T@qw; qn=qw[0]; a_nor=a[0]; j=1+int(np.argmax(a[1:])); a_mal=a[j]
    m=min(1.0,max(0.0,(a_nor-a_mal-2*(1-qn))/(qn-2)))
    p=np.zeros(7); p[0]=1-m; p[j]=m
    return p, sum(q[t]*harm_score(p,t) for t in range(7))

def decode_batch(Q, w=None):
    P=np.zeros_like(Q,float); Eh=np.zeros(len(Q))
    for i in range(len(Q)): P[i],Eh[i]=decode_belief(Q[i],w)
    return P, Eh

def referral_policy(Eh, frac):
    Eh=np.asarray(Eh,float); N=len(Eh); upl=S_REF-Eh; order=np.argsort(-upl)
    k=int(round(frac*N)); ref=np.zeros(N); sel=[i for i in order[:k] if upl[i]>0]
    for r,i in enumerate(sel):
        ref[i]= (1.0-0.45*(r/max(1,len(sel)-1))) if len(sel)>1 else 0.9
    return ref

def tune_frac_and_score(Q, gt, grid=FRAC_GRID, w=None):
    P,Eh=decode_batch(Q,w); best=(-1,0.0,None)
    for fr in grid:
        ref=referral_policy(Eh,fr); s=exact_grade(P,ref,gt)
        if s>best[0]: best=(s,fr,ref)
    return best  # (score, frac, ref)

# ----------------------------- data -----------------------------
def find_data():
    import zipfile
    cands = [Path("./dataset/public"), Path("../dataset/public"),
             Path("G:/ml/data/runnerloaddataset/public")]
    if Path("/kaggle/input").exists():
        cands += sorted(Path("/kaggle/input").glob("*"))
        cands += sorted(Path("/kaggle/input").glob("*/*"))
    for c in cands:
        if (c/"train.csv").exists() and (c/"images").exists(): return c
    roots = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else []
    # extracted at any depth
    for root in roots:
        for t in root.rglob("train.csv"):
            if (t.parent/"images").exists(): return t.parent
    # zip that needs extraction
    for root in roots:
        for z in sorted(root.rglob("*.zip")):
            dest = Path("/kaggle/working/_data"); dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(z) as zf: zf.extractall(dest)
            if (dest/"train.csv").exists() and (dest/"images").exists(): return dest
            for t in dest.rglob("train.csv"):
                if (t.parent/"images").exists(): return t.parent
    for root in roots:
        for p in list(root.rglob("*"))[:50]: print("INPUT:", p)
    raise FileNotFoundError("could not locate dataset (train.csv + images/)")
DATA = find_data()
print("DATA:", DATA, "| device:", DEV, "| backbone:", BACKBONE)

cfg = timm.get_pretrained_cfg(BACKBONE) if not SMOKE else None
MEAN = list(cfg.mean) if cfg else [0.485,0.456,0.406]
STD  = list(cfg.std)  if cfg else [0.229,0.224,0.225]

def hed_jitter(arr, sigma=0.04):  # arr float[0,1] HWC RGB
    from skimage.color import rgb2hed, hed2rgb
    hed = rgb2hed(arr)
    for c in range(3):
        hed[...,c] = hed[...,c]*(1+np.random.uniform(-sigma,sigma)) + np.random.uniform(-sigma,sigma)
    return np.clip(hed2rgb(hed),0,1)

import torchvision.transforms.functional as TF
class DS(Dataset):
    def __init__(self, df, train):
        self.df=df.reset_index(drop=True); self.train=train
    def __len__(self): return len(self.df)
    def _load(self, name):
        im=Image.open(DATA/"images"/name).convert("RGB")
        # pre-shrink longer side to ~512 for speed while keeping detail
        im.thumbnail((512,512)); return im
    def __getitem__(self,i):
        r=self.df.iloc[i]; im=self._load(r["image"]); mag=0 if str(r["magnification"]).startswith("20") else 1
        if self.train:
            im=TF.resize(im,(288,288))
            im=TF.to_tensor(im)
            # random resized crop
            import torchvision.transforms as T
            i0,j0,h,w=T.RandomResizedCrop.get_params(im,scale=(0.45,1.0),ratio=(0.8,1.25))
            im=TF.resized_crop(im,i0,j0,h,w,(IMG,IMG))
            if random.random()<0.5: im=TF.hflip(im)
            if random.random()<0.5: im=TF.vflip(im)
            k=random.randint(0,3); im=torch.rot90(im,k,[1,2]) if k else im
            arr=im.permute(1,2,0).numpy()
            if random.random()<0.6: arr=hed_jitter(arr)
            im=torch.from_numpy(arr).permute(2,0,1).float()
            im=TF.normalize(im,MEAN,STD)
        else:
            im=TF.resize(im,(IMG,IMG)); im=TF.to_tensor(im); im=TF.normalize(im,MEAN,STD)
        y=CIDX[r["label"]] if "label" in r and isinstance(r["label"],str) else -1
        return im, mag, y

# ----------------------------- model -----------------------------
def make_backbone(backbone):
    """Load pretrained backbone offline if weights are staged, else download (dev)."""
    for wd in [os.environ.get("WEIGHTS_DIR",""), "/kaggle/input/timm-weights", "./weights"]:
        if wd and Path(wd).exists():
            cks=sorted(list(Path(wd).rglob("*.safetensors"))+list(Path(wd).rglob("*.bin"))+list(Path(wd).rglob("*.pth")))
            for ck in cks:
                try:
                    m=timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
                    timm.models.load_checkpoint(m, str(ck), strict=False); print("offline weights:", ck.name); return m
                except Exception as e: print("skip", ck.name, str(e)[:80])
    return timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="avg")

class Net(nn.Module):
    def __init__(self, backbone, mag_dim=8, drop=0.3):
        super().__init__()
        self.bb=make_backbone(backbone)
        f=self.bb.num_features
        self.mag=nn.Embedding(2, mag_dim)
        self.head=nn.Sequential(nn.Dropout(drop), nn.Linear(f+mag_dim, 7))
        self.bin_head=nn.Linear(f+mag_dim, 1)   # aux benign<->malignant axis (guide: dominant)
    def forward(self,x,mag):
        z=torch.cat([self.bb(x), self.mag(mag)],1)
        return self.head(z), self.bin_head(z).squeeze(1)

class EMA:
    def __init__(self,model,decay):
        self.decay=decay; self.t=0; self.shadow={k:v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self,model):
        self.t+=1; d=min(self.decay,(1+self.t)/(10+self.t))   # warmup: track closely early, avoids random-head leak
        for k,v in model.state_dict().items():
            if v.dtype.is_floating_point: self.shadow[k].mul_(d).add_(v.detach(),alpha=1-d)
            else: self.shadow[k]=v.detach().clone()
    def copy_to(self,model): model.load_state_dict(self.shadow,strict=True)

@torch.no_grad()
def predict_logits(model, df, tta):
    model.eval(); dl=DataLoader(DS(df,False),batch_size=32,shuffle=False,num_workers=2 if DEV=="cuda" else 0)
    outs=[]
    for im,mag,_ in dl:
        im=im.to(DEV); mag=mag.to(DEV)
        lo=model(im,mag)[0]
        if tta:
            lo=lo+model(torch.flip(im,[3]),mag)[0]+model(torch.flip(im,[2]),mag)[0]; lo=lo/3
        outs.append(lo.float().cpu())
    return torch.cat(outs).numpy()

def fit_temperature(logits, labels):
    lg=torch.tensor(logits,dtype=torch.float32); lb=torch.tensor(labels,dtype=torch.long)
    T=torch.nn.Parameter(torch.ones(1)); opt=torch.optim.LBFGS([T],lr=0.1,max_iter=60)
    def closure(): opt.zero_grad(); l=F.cross_entropy(lg/T.clamp(min=0.05),lb); l.backward(); return l
    opt.step(closure); return float(T.clamp(min=0.05).item())

def softmax_np(z,T=1.0):
    z=z/T; z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

# ----------------------------- train one fold -----------------------------
def train_fold(tr_df, va_df, class_w):
    set_seed(SEED)
    model=Net(BACKBONE).to(DEV); ema=EMA(model,EMA_DECAY)
    bb=[p for n,p in model.named_parameters() if n.startswith("bb.")]
    hd=[p for n,p in model.named_parameters() if not n.startswith("bb.")]
    opt=torch.optim.AdamW([{"params":bb,"lr":LR*BB_LR_MULT},{"params":hd,"lr":LR}],weight_decay=WD)
    dl=DataLoader(DS(tr_df,True),batch_size=BS,shuffle=True,drop_last=False,
                  num_workers=2 if DEV=="cuda" else 0,pin_memory=DEV=="cuda")
    steps=max(1,len(dl))*EPOCHS
    sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[LR*BB_LR_MULT,LR],total_steps=steps,pct_start=0.15)
    scaler=torch.cuda.amp.GradScaler(enabled=DEV=="cuda")
    cw=torch.tensor(class_w,dtype=torch.float32,device=DEV)
    for ep in range(EPOCHS):
        for p in model.bb.parameters(): p.requires_grad = (ep>=FREEZE_EP)  # freeze backbone for head warmup
        model.train()
        for im,mag,y in dl:
            im=im.to(DEV); mag=mag.to(DEV); y=y.to(DEV)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=DEV=="cuda"):
                out,binout=model(im,mag); loss=F.cross_entropy(out,y,weight=cw,label_smoothing=LS)
                if LAMBDA_BIN>0:
                    loss=loss+LAMBDA_BIN*F.binary_cross_entropy_with_logits(binout,(y>0).float())
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); ema.update(model)
    em=Net(BACKBONE).to(DEV); ema.copy_to(em)
    va_logits=predict_logits(em, va_df, TTA)
    return em, va_logits

# ----------------------------- main -----------------------------
def main():
    from sklearn.model_selection import StratifiedGroupKFold
    tr=pd.read_csv(DATA/"train.csv"); te=pd.read_csv(DATA/"test.csv")
    if SMOKE:
        tr=tr.groupby("label",group_keys=False).head(3).reset_index(drop=True)
        te=te.head(6).reset_index(drop=True)
        globals()["EPOCHS"]=1; globals()["N_SPLITS"]=2
    y=tr["label"].map(CIDX).values; groups=tr["patient_id"].values
    cnt=np.bincount(y,minlength=7); class_w=(len(y)/(7*np.maximum(cnt,1))); class_w=class_w/class_w.mean()
    print("counts",dict(zip(CLASSES,cnt)),"\nclass_w",np.round(class_w,2))

    oof_logits=np.zeros((len(tr),7)); test_logits_folds=[]
    for seed in SEEDS:
        sgkf=StratifiedGroupKFold(n_splits=N_SPLITS,shuffle=True,random_state=seed)
        for f,(tri,vai) in enumerate(sgkf.split(tr,y,groups)):
            t0=time.time()
            em,va_lo=train_fold(tr.iloc[tri], tr.iloc[vai], class_w)
            oof_logits[vai]=va_lo
            test_logits_folds.append(predict_logits(em, te, TTA))
            del em; torch.cuda.empty_cache() if DEV=="cuda" else None
            acc=(va_lo.argmax(1)==y[vai]).mean()
            print(f"  seed{seed} fold{f}: val_acc={acc:.3f}  ({time.time()-t0:.0f}s)")

    # calibrate on OOF
    T=fit_temperature(oof_logits, y); print("temperature T=",round(T,3))
    oofQ=softmax_np(oof_logits,T)
    # macro inverse-freq weights for decode (estimated from train prior)
    w_macro=(1.0/np.maximum(cnt,1)); w_macro=w_macro/w_macro.mean()

    # evaluate: raw vs decode vs decode+macroweight, tune frac
    s_raw=exact_grade(oofQ, np.zeros(len(y)), y)
    s_dec,fr_dec,_=tune_frac_and_score(oofQ,y,w=None)
    s_decw,fr_decw,_=tune_frac_and_score(oofQ,y,w=w_macro)
    use_w = w_macro if s_decw>=s_dec else None
    best_s, best_fr, _ = (s_decw,fr_decw,None) if s_decw>=s_dec else (s_dec,fr_dec,None)
    oof_acc=(oofQ.argmax(1)==y).mean(); oof_bacc=np.mean([ (oofQ[y==c].argmax(1)==c).mean() for c in range(7) if (y==c).any()])
    # diagnostics: per-class recall, dominant benign<->malignant axis, decode peakiness
    pred=oofQ.argmax(1)
    per_class={CLASSES[c]: round(float((pred[y==c]==c).mean()),3) for c in range(7) if (y==c).any()}
    is_mal=(y!=0).astype(int); pred_mal=(oofQ[:,1:].sum(1)>0.5).astype(int)
    bin_acc=float((pred_mal==is_mal).mean())
    miss=float(((is_mal==1)&(pred_mal==0)).sum()/max(1,(is_mal==1).sum()))   # malignant called benign
    fa=float(((is_mal==0)&(pred_mal==1)).sum()/max(1,(is_mal==0).sum()))     # benign called malignant
    Pdec,_=decode_batch(oofQ, use_w); peak=float(Pdec.max(1).mean())
    print(f"\nOOF acc={oof_acc:.3f} balanced_acc={oof_bacc:.3f}")
    print(f"per-class recall: {per_class}")
    print(f"benign/malignant: bin_acc={bin_acc:.3f}  miss(mal->ben)={miss:.3f}  false_alarm(ben->mal)={fa:.3f}")
    print(f"decode peakiness (mean max-prob)={peak:.3f}")
    print(f"OOF grade: raw_submit={s_raw:.4f} | decode={s_dec:.4f}(fr={fr_dec}) | "
          f"decode+macroW={s_decw:.4f}(fr={fr_decw}) | BEST={best_s:.4f}")

    # ---- test inference (fold ensemble) ----
    test_logits=np.mean(np.stack(test_logits_folds),0)
    teQ=softmax_np(test_logits,T)
    P,Eh=decode_batch(teQ, use_w); ref=referral_policy(Eh,best_fr)
    sub=pd.DataFrame({"id":te["id"]})
    for j,c in enumerate(CLASSES): sub["p_"+c]=P[:,j]
    sub["referral"]=ref
    sub.to_csv(WORK/"submission.csv",index=False)
    # save logits for offline ensembling / decode-temper experiments
    np.savez(WORK/"oof.npz", oof_logits=oof_logits, test_logits=test_logits,
             y=y, ids=te["id"].values.astype(str), T=np.float32(T), backbone=BACKBONE)
    json.dump({"oof_score":best_s,"oof_acc":float(oof_acc),"oof_bacc":float(oof_bacc),
               "temperature":T,"frac":best_fr,"macro_w":bool(use_w is not None),
               "backbone":BACKBONE,"raw":s_raw,"decode":s_dec,"decode_w":s_decw,
               "per_class_recall":per_class,"bin_acc":bin_acc,"miss":miss,"false_alarm":fa,
               "decode_peakiness":peak,"epochs":EPOCHS,"n_splits":N_SPLITS,"seeds":SEEDS},
              open(WORK/"oof_metrics.json","w"), indent=2)
    print("\nwrote", WORK/"submission.csv", sub.shape)
    print(sub.head(3).to_string())

if __name__=="__main__":
    main()
