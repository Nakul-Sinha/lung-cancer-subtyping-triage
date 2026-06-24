"""
Lung Cancer Subtyping & Grading — official self-contained solution.
Reads dataset/public (auto-detected), writes ./working/submission.csv.

Design: the score on this metric is dominated by the decision layer, not raw accuracy.
This pipeline is SELF-CALIBRATING and robust to data quality:
  1. Always builds a magnification-conditioned class prior (a legitimate test-available covariate).
  2. If a GPU is present, also fine-tunes an ImageNet backbone (with a magnification covariate).
  3. On patient-grouped OOF it scores every candidate posterior (prior / vision / blends) with the
     EXACT competition grader, and keeps whichever maximizes the real metric.
  4. Converts the chosen calibrated posterior into the Bayes-optimal belief for this (non-proper)
     metric and adds a budget-aware, expected-harm referral policy.
So if the images carry signal the vision model wins automatically; if not, it floors on the
magnification prior — never below a strong post-processing baseline.
"""
import os, json, math, random, warnings
from pathlib import Path
from math import ceil
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED); np.random.seed(SEED)
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
DISABLE_VISION = os.environ.get("DISABLE_VISION", "0") == "1"
FORCE_VISION = os.environ.get("FORCE_VISION", "0") == "1"   # run vision path even on CPU (smoke)
SMOKE = os.environ.get("SMOKE", "0") == "1"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./working")
WORK.mkdir(exist_ok=True, parents=True)

# ----------------------------- metric / decode (verified gap=0 vs brute force) -----------------------------
CLASSES = ["nor","aca_bd","aca_md","aca_pd","scc_bd","scc_md","scc_pd"]
CIDX = {c:i for i,c in enumerate(CLASSES)}; MALIG = np.array([0,1,1,1,1,1,1], bool)
GAMMA_MISS, GAMMA_FA, S_REF = 1.0, 0.5, 0.40
REF_BUDGET, REF_THR, HARM_FLOOR, HARM_CEIL, SCORE_FLOOR = 0.20, 0.5, -2.0, 1.0, 0.02
def _sub(c): return c.split("_")[0]
def _gr(c): return {"bd":1,"md":2,"pd":3}.get(c.split("_")[1],0) if "_" in c else 0
def _aff(t,k):
    if k==t: return 1.0
    if t=="nor" or k=="nor": return -1.0
    if _sub(t)!=_sub(k): return 0.0
    return {1:0.5,2:0.2}[abs(_gr(t)-_gr(k))]
A = np.array([[_aff(t,k) for k in CLASSES] for t in CLASSES])
def harm_score(p,t):
    p=np.asarray(p,float); hr=float(p@A[t]); pen=GAMMA_MISS*p[0]**2 if t!=0 else GAMMA_FA*(p[MALIG].sum())**2
    return max(HARM_FLOOR,min(HARM_CEIL,hr-pen))
def exact_grade(P,ref,gt):
    P=np.asarray(P,float); ref=np.asarray(ref,float); gt=np.asarray(gt,int); N=len(gt)
    K=int(REF_BUDGET*N); cap=max(1,ceil(K/7))
    elig=sorted([i for i in range(N) if ref[i]>REF_THR], key=lambda i:(-ref[i],i))
    referred=set(); cc=[0]*7
    for i in elig:
        if len(referred)>=K: break
        if cc[gt[i]]<cap: referred.add(i); cc[gt[i]]+=1
    by={c:[] for c in range(7)}
    for i in range(N): by[gt[i]].append(S_REF if i in referred else harm_score(P[i],gt[i]))
    macro=float(np.mean([np.mean(v) for v in by.values() if v]))
    return max(SCORE_FLOOR,min(1.0,SCORE_FLOOR+(macro-HARM_FLOOR)/(HARM_CEIL-HARM_FLOOR)*(1.0-SCORE_FLOOR)))
def decode_belief(q,w=None):
    q=np.asarray(q,float); qw=q*(np.ones(7) if w is None else w); qw=qw/qw.sum()
    a=A.T@qw; qn=qw[0]; a_nor=a[0]; j=1+int(np.argmax(a[1:])); a_mal=a[j]
    m=min(1.0,max(0.0,(a_nor-a_mal-2*(1-qn))/(qn-2))); p=np.zeros(7); p[0]=1-m; p[j]=m
    return p, sum(q[t]*harm_score(p,t) for t in range(7))
def decode_batch(Q,w=None):
    P=np.zeros_like(Q,float); Eh=np.zeros(len(Q))
    for i in range(len(Q)): P[i],Eh[i]=decode_belief(Q[i],w)
    return P,Eh
def referral_policy(Eh,frac):
    Eh=np.asarray(Eh,float); N=len(Eh); upl=S_REF-Eh; order=np.argsort(-upl)
    k=int(round(frac*N)); ref=np.zeros(N); sel=[i for i in order[:k] if upl[i]>0]
    for r,i in enumerate(sel): ref[i]=(1.0-0.45*(r/max(1,len(sel)-1))) if len(sel)>1 else 0.9
    return ref
def best_decode(Q, y, cnt):
    """tune {macro-weight exponent, referral-frac} on the exact grader; return (score, cfg, apply).
    macro-weight (1/cnt)^exp reweights affinity toward rare classes (macro-average aware)."""
    best=(-1,None,None)
    for exp in [0.0,0.5,1.0,1.5,2.0]:
        w = None if exp==0 else (lambda v: v/v.mean())((1.0/np.maximum(cnt,1))**exp)
        P,Eh=decode_batch(Q,w)
        for fr in [0,.1,.15,.2,.25,.3,.35,.4]:
            s=exact_grade(P,referral_policy(Eh,fr),y)
            if s>best[0]: best=(s,dict(exp=exp,frac=fr),w)
    s,cfg,wsel=best
    def apply(testQ):
        P,Eh=decode_batch(testQ,wsel); return P, referral_policy(Eh,cfg["frac"])
    return s, cfg, apply

# ----------------------------- data -----------------------------
def find_data():
    import zipfile
    cands=[Path("./dataset/public"),Path("../dataset/public"),
           Path("G:/Datacurve/eris/lung-cancer-dataset"),Path("G:/Datacurve/eris/runnerloaddataset/public")]
    if Path("/kaggle/input").exists():
        cands+=sorted(Path("/kaggle/input").glob("*")); cands+=sorted(Path("/kaggle/input").glob("*/*"))
    for c in cands:
        if (c/"train.csv").exists() and (c/"images").exists(): return c
    roots=[Path("/kaggle/input")] if Path("/kaggle/input").exists() else []
    for root in roots:
        for t in root.rglob("train.csv"):
            if (t.parent/"images").exists(): return t.parent
    raise FileNotFoundError("dataset (train.csv + images/) not found")

def mag_bin(s): return 0 if str(s).startswith("20") else 1

def grouped_folds(y, groups, seed=SEED):
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf=StratifiedGroupKFold(N_SPLITS,shuffle=True,random_state=seed)
    fold=np.full(len(y),-1)
    for f,(_,vai) in enumerate(sgkf.split(np.zeros(len(y)),y,groups)): fold[vai]=f
    return fold

def prior_oof_and_test(y, magtr, fold, magte, alpha=1.0):
    """magnification-conditioned class prior via grouped OOF; and test posteriors from full train."""
    oof=np.zeros((len(y),7))
    for f in range(N_SPLITS):
        tri=fold!=f; vai=fold==f
        for mg in [0,1]:
            s=tri&(magtr==mg)
            pr=np.bincount(y[s],minlength=7)+alpha; pr=pr/pr.sum()
            oof[vai&(magtr==mg)]=pr
    test=np.zeros((len(magte),7))
    for mg in [0,1]:
        pr=np.bincount(y[magtr==mg],minlength=7)+alpha; pr=pr/pr.sum()
        test[magte==mg]=pr
    return oof, test

# ----------------------------- optional vision model (GPU) -----------------------------
def vision_oof_and_test(tr, te, y, fold, DATA):
    import torch, torch.nn as nn, torch.nn.functional as F, timm
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    import torchvision.transforms.functional as TF
    DEV="cuda" if torch.cuda.is_available() else "cpu"
    AMP = (DEV=="cuda")
    BACKBONE=os.environ.get("BACKBONE","convnext_tiny.fb_in22k_ft_in1k")
    IMG=224; EPOCHS=int(os.environ.get("EPOCHS","24")); BS=int(os.environ.get("BS","16"))
    LR=float(os.environ.get("LR","2e-4")); WD=0.05; LS=0.05; BBM=float(os.environ.get("BB_LR_MULT","0.3")); FREEZE=int(os.environ.get("FREEZE_EP","3"))
    cfg=timm.get_pretrained_cfg(BACKBONE); MEAN,STD=list(cfg.mean),list(cfg.std)
    def hed(arr,s=0.04):
        from skimage.color import rgb2hed,hed2rgb; h=rgb2hed(arr)
        for c in range(3): h[...,c]=h[...,c]*(1+np.random.uniform(-s,s))+np.random.uniform(-s,s)
        return np.clip(hed2rgb(h),0,1)
    class DS(Dataset):
        def __init__(s,df,tr): s.df=df.reset_index(drop=True); s.tr=tr
        def __len__(s): return len(s.df)
        def __getitem__(s,i):
            r=s.df.iloc[i]; im=Image.open(DATA/"images"/r["image"]).convert("RGB"); im.thumbnail((512,512))
            mg=mag_bin(r["magnification"])
            if s.tr:
                im=TF.resize(im,(288,288)); im=TF.to_tensor(im)
                import torchvision.transforms as T
                i0,j0,h,w=T.RandomResizedCrop.get_params(im,scale=(0.45,1.0),ratio=(0.8,1.25))
                im=TF.resized_crop(im,i0,j0,h,w,(IMG,IMG))
                if random.random()<0.5: im=TF.hflip(im)
                if random.random()<0.5: im=TF.vflip(im)
                k=random.randint(0,3); im=torch.rot90(im,k,[1,2]) if k else im
                a=im.permute(1,2,0).numpy()
                if random.random()<0.6: a=hed(a)
                im=TF.normalize(torch.from_numpy(a).permute(2,0,1).float(),MEAN,STD)
            else:
                im=TF.normalize(TF.to_tensor(TF.resize(im,(IMG,IMG))),MEAN,STD)
            yy=CIDX[r["label"]] if "label" in r and isinstance(r.get("label"),str) else -1
            return im, mg, yy
    def make_bb():
        for wd in [os.environ.get("WEIGHTS_DIR",""),"/kaggle/input/timm-weights","./weights"]:
            if wd and Path(wd).exists():
                for ck in sorted(list(Path(wd).rglob("*.safetensors"))+list(Path(wd).rglob("*.bin"))):
                    try:
                        m=timm.create_model(BACKBONE,pretrained=False,num_classes=0,global_pool="avg")
                        timm.models.load_checkpoint(m,str(ck),strict=False); return m
                    except Exception: pass
        return timm.create_model(BACKBONE,pretrained=True,num_classes=0,global_pool="avg")
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.bb=make_bb(); f=s.bb.num_features
            s.mag=nn.Embedding(2,8); s.head=nn.Sequential(nn.Dropout(0.3),nn.Linear(f+8,7))
        def forward(s,x,mg): return s.head(torch.cat([s.bb(x),s.mag(mg)],1))
    class EMA:
        def __init__(s,m,d=0.999): s.d=d; s.t=0; s.sh={k:v.detach().clone() for k,v in m.state_dict().items()}
        def upd(s,m):
            s.t+=1; d=min(s.d,(1+s.t)/(10+s.t))
            for k,v in m.state_dict().items():
                s.sh[k]=(s.sh[k].mul_(d).add_(v.detach(),alpha=1-d)) if v.dtype.is_floating_point else v.detach().clone()
        def to(s,m): m.load_state_dict(s.sh,strict=True)
    NW = 2 if DEV=="cuda" else 0
    @torch.no_grad()
    def pred(m,df):
        m.eval(); dl=DataLoader(DS(df,False),batch_size=32,num_workers=NW)
        out=[]
        for im,mg,_ in dl:
            im=im.to(DEV); mg=mg.to(DEV); lo=m(im,mg)+m(torch.flip(im,[3]),mg)+m(torch.flip(im,[2]),mg)
            out.append((lo/3).float().cpu().numpy())
        return np.concatenate(out,0)
    cnt=np.bincount(y,minlength=7); cw=torch.tensor(len(y)/(7*np.maximum(cnt,1)),dtype=torch.float32,device=DEV); cw/=cw.mean()
    oof=np.zeros((len(y),7)); tests=[]
    for f in range(N_SPLITS):
        tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]
        torch.manual_seed(SEED); m=Net().to(DEV); ema=EMA(m)
        bb=[p for n,p in m.named_parameters() if n.startswith("bb.")]; hd=[p for n,p in m.named_parameters() if not n.startswith("bb.")]
        opt=torch.optim.AdamW([{"params":bb,"lr":LR*BBM},{"params":hd,"lr":LR}],weight_decay=WD)
        dl=DataLoader(DS(tr.iloc[tri],True),batch_size=BS,shuffle=True,num_workers=NW,pin_memory=AMP)
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[LR*BBM,LR],total_steps=max(1,len(dl))*EPOCHS,pct_start=0.15)
        sc=torch.cuda.amp.GradScaler(enabled=AMP)
        for ep in range(EPOCHS):
            for p in m.bb.parameters(): p.requires_grad=(ep>=FREEZE)
            m.train()
            for im,mg,yy in dl:
                im=im.to(DEV);mg=mg.to(DEV);yy=yy.to(DEV); opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=AMP):
                    loss=F.cross_entropy(m(im,mg),yy,weight=cw,label_smoothing=LS)
                sc.scale(loss).backward(); sc.step(opt); sc.update(); sch.step(); ema.upd(m)
        em=Net().to(DEV); ema.to(em)
        oof[vai]=pred(em,tr.iloc[vai]); tests.append(pred(em,te))
        del m,em; torch.cuda.empty_cache()
        print(f"  vision fold{f} val_acc={(oof[vai].argmax(1)==y[vai]).mean():.3f}")
    # temperature calibration on OOF
    import torch as T2
    lg=T2.tensor(oof,dtype=T2.float32); lb=T2.tensor(y)
    t=T2.nn.Parameter(T2.ones(1)); o=T2.optim.LBFGS([t],lr=0.1,max_iter=60)
    def cl(): o.zero_grad(); l=F.cross_entropy(lg/t.clamp(min=.05),lb); l.backward(); return l
    o.step(cl); Tp=float(t.clamp(min=.05))
    def sm(z,Tt): z=z/Tt; z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
    return sm(oof,Tp), sm(np.mean(tests,0),Tp)

# ----------------------------- main -----------------------------
def main():
    DATA=find_data(); print("DATA:",DATA)
    tr=pd.read_csv(DATA/"train.csv"); te=pd.read_csv(DATA/"test.csv")
    if SMOKE:
        tr=tr.groupby("label",group_keys=False).head(6).reset_index(drop=True); te=te.head(10).reset_index(drop=True)
        os.environ["EPOCHS"]="1"; globals()["N_SPLITS"]=2
    y=tr["label"].map(CIDX).values; groups=tr["patient_id"].values
    magtr=np.array([mag_bin(s) for s in tr["magnification"]]); magte=np.array([mag_bin(s) for s in te["magnification"]])
    cnt=np.bincount(y,minlength=7)
    fold=grouped_folds(y,groups)

    cands={}  # name -> (oofQ, testQ)
    pr_oof, pr_te = prior_oof_and_test(y,magtr,fold,magte); cands["mag_prior"]=(pr_oof,pr_te)

    import torch
    use_vision = (torch.cuda.is_available() or FORCE_VISION) and not DISABLE_VISION
    if use_vision:
        try:
            v_oof,v_te=vision_oof_and_test(tr,te,y,fold,DATA); cands["vision"]=(v_oof,v_te)
            for a in (0.5,0.75):
                cands[f"blend{a}"]=(a*v_oof+(1-a)*pr_oof, a*v_te+(1-a)*pr_te)
        except Exception as e:
            print("vision path skipped:",str(e)[:200])
    else:
        print("no GPU (or disabled) -> magnification-prior path only")

    print(f"\n{'candidate':12s} {'val_acc':>7} {'OOF_grade':>9}  cfg")
    scored={}
    for name,(oq,tq) in cands.items():
        acc=(oq.argmax(1)==y).mean(); s,cfg,apply=best_decode(oq,y,cnt)
        print(f"{name:12s} {acc:7.3f} {s:9.4f}  {cfg}")
        scored[name]=(s,cfg,apply)
    # Robust selection: magnification-prior is the safe default (stable, can't overfit).
    # Only switch to a learned candidate if it beats the prior by more than CV noise.
    MARGIN=float(os.environ.get("SELECT_MARGIN","0.01"))
    base_s,base_cfg,base_apply=scored["mag_prior"]
    name,(s,cfg,apply)="mag_prior",(base_s,base_cfg,base_apply)
    for cand in [c for c in scored if c!="mag_prior"]:
        cs,ccfg,capply=scored[cand]
        if cs>s+MARGIN: name,(s,cfg,apply)=cand,(cs,ccfg,capply)
    print(f"\nSELECTED: {name}  OOF_grade={s:.4f}  {cfg}  (prior={base_s:.4f}, margin={MARGIN})")

    P,ref=apply(cands[name][1])
    sub=pd.DataFrame({"id":te["id"]})
    for j,c in enumerate(CLASSES): sub["p_"+c]=P[:,j]
    sub["referral"]=ref
    sub.to_csv(WORK/"submission.csv",index=False)
    json.dump({"selected":name,"oof_grade":s,"cfg":cfg,"candidates":{k:float(best_decode(v[0],y,cnt)[0]) for k,v in cands.items()}},
              open(WORK/"oof_metrics.json","w"),indent=2)
    print("wrote",WORK/"submission.csv",sub.shape)

if __name__=="__main__": main()
