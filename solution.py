import os, json, warnings
from pathlib import Path
from math import ceil
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
N_SEEDS = int(os.environ.get("N_SEEDS", "10"))
ALPHA = float(os.environ.get("ALPHA", "0.3"))
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./working")
WORK.mkdir(exist_ok=True, parents=True)

CLASSES = ["nor","aca_bd","aca_md","aca_pd","scc_bd","scc_md","scc_pd"]
CIDX = {c:i for i,c in enumerate(CLASSES)}
MALIG = np.array([0,1,1,1,1,1,1], bool)
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
    p=np.asarray(p,float); hr=float(p@A[t])
    pen=GAMMA_MISS*p[0]**2 if t!=0 else GAMMA_FA*(p[MALIG].sum())**2
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
    m=min(1.0,max(0.0,(a_nor-a_mal-2*(1-qn))/(qn-2)))
    p=np.zeros(7); p[0]=1-m; p[j]=m
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

def macro_weight(exp, cnt):
    if exp==0: return None
    v=(1.0/np.maximum(cnt,1))**exp; return v/v.mean()

def find_data():
    cands=[Path("./dataset/public"),Path("../dataset/public"),
           Path("G:/Datacurve/eris/lung-cancer-dataset"),Path("G:/Datacurve/eris/runnerloaddataset/public")]
    if Path("/kaggle/input").exists():
        cands+=sorted(Path("/kaggle/input").glob("*")); cands+=sorted(Path("/kaggle/input").glob("*/*"))
    for c in cands:
        if (c/"train.csv").exists() and (c/"images").exists(): return c
    if Path("/kaggle/input").exists():
        for t in Path("/kaggle/input").rglob("train.csv"):
            if (t.parent/"images").exists(): return t.parent
    raise FileNotFoundError("dataset (train.csv + images/) not found")

def mag_bin(s): return 0 if str(s).startswith("20") else 1

def magnification_prior(y_tr, mag_tr, mag_target):
    out=np.zeros((len(mag_target),7))
    for mg in [0,1]:
        pr=np.bincount(y_tr[mag_tr==mg],minlength=7)+ALPHA; pr=pr/pr.sum()
        out[mag_target==mg]=pr
    return out

def oof_posteriors(y, mag, seed):
    from sklearn.model_selection import StratifiedKFold
    oof=np.zeros((len(y),7))
    for tri,vai in StratifiedKFold(N_SPLITS,shuffle=True,random_state=seed).split(np.zeros(len(y)),y):
        oof[vai]=magnification_prior(y[tri], mag[tri], mag[vai])
    return oof

def tune_config(y, mag, cnt):
    oofs=[oof_posteriors(y, mag, sd) for sd in range(N_SEEDS)]
    best=(-1,None)
    for exp in [0.0,0.5,1.0,1.5,2.0]:
        w=macro_weight(exp,cnt)
        decoded=[decode_batch(o,w) for o in oofs]
        for frac in [0.0,0.2,0.3,0.4,0.5,0.6]:
            sc=np.mean([exact_grade(P,referral_policy(Eh,frac),y) for P,Eh in decoded])
            if sc>best[0]: best=(sc,dict(exp=exp,frac=frac))
    return best

def main():
    DATA=find_data(); print("DATA:",DATA)
    tr=pd.read_csv(DATA/"train.csv"); te=pd.read_csv(DATA/"test.csv")
    y=tr["label"].map(CIDX).values
    mag_tr=np.array([mag_bin(s) for s in tr["magnification"]])
    mag_te=np.array([mag_bin(s) for s in te["magnification"]])
    cnt=np.bincount(y,minlength=7)

    score,cfg=tune_config(y, mag_tr, cnt)
    print(f"shared-pool OOF={score:.4f}  config={cfg}")

    w=macro_weight(cfg["exp"],cnt)
    teQ=magnification_prior(y, mag_tr, mag_te)
    P,Eh=decode_batch(teQ,w); ref=referral_policy(Eh,cfg["frac"])
    sub=pd.DataFrame({"id":te["id"]})
    for j,c in enumerate(CLASSES): sub["p_"+c]=P[:,j]
    sub["referral"]=ref
    sub.to_csv(WORK/"submission.csv",index=False)
    json.dump({"oof_score":score,"config":cfg,"alpha":ALPHA,"n_seeds":N_SEEDS,
               "n_referred_flagged":int((ref>0.5).sum())}, open(WORK/"oof_metrics.json","w"), indent=2)
    print("wrote",WORK/"submission.csv",sub.shape)

if __name__=="__main__": main()
