"""Headless Kaggle T4 driver for the lung-cancer solution (adapted from data/kaggle_run).
  python dev/krun.py dataset   # zip public data -> create/version private Kaggle dataset
  python dev/krun.py push      # push the GPU script kernel (train.py)
  python dev/krun.py status    # poll kernel status
  python dev/krun.py fetch     # download submission.csv + oof_metrics.json + log
"""
import json, shutil, sys, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = Path("G:/ml/data/runnerloaddataset/public")
STAGE = HERE/"kdataset"
SRC = HERE/"train.py"
import os
DATASET_SLUG = "lung-cancer-subtyping-public"
KERNEL_SLUG  = os.environ.get("KSLUG", "lung-cancer-subtyping-grading")
KERNEL_DIR = HERE/("kkernel_"+KERNEL_SLUG); OUT_DIR = HERE/("kout_"+KERNEL_SLUG)
ENV = {k: v for k, v in os.environ.items()
       if k in ("BACKBONE","EPOCHS","BS","N_SPLITS","SEEDS","LR","WD","IMG","TTA","FRAC_GRID","LS","EMA","LAMBDA_BIN")}

def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi(); a.authenticate(); return a
def user(a):
    u = (getattr(a,"config_values",{}) or {}).get("username")
    if not u: raise RuntimeError("no kaggle username");
    return u

def do_dataset():
    a=api(); u=user(a); STAGE.mkdir(parents=True,exist_ok=True)
    z=STAGE/"data.zip"
    if z.exists(): z.unlink()
    files=[p for p in DATA_DIR.rglob("*") if p.is_file() and p.name!="dataset-metadata.json"]
    with zipfile.ZipFile(z,"w",zipfile.ZIP_DEFLATED) as zf:
        for p in files: zf.write(p, p.relative_to(DATA_DIR).as_posix())
    print(f"zipped {len(files)} files -> {z} ({z.stat().st_size//(1024*1024)} MB)")
    meta={"title":"lung cancer subtyping public","id":f"{u}/{DATASET_SLUG}","licenses":[{"name":"CC0-1.0"}]}
    (STAGE/"dataset-metadata.json").write_text(json.dumps(meta,indent=2))
    try:
        a.dataset_create_new(str(STAGE),public=False,quiet=False,convert_to_csv=False,dir_mode="skip")
        print("dataset created:",meta["id"])
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            a.dataset_create_version(str(STAGE),version_notes="update",quiet=False,convert_to_csv=False,dir_mode="skip")
            print("dataset version updated:",meta["id"])
        else: raise

def do_push():
    a=api(); u=user(a); KERNEL_DIR.mkdir(parents=True,exist_ok=True)
    # inject env overrides at the top of the script so the kernel uses our config
    code=SRC.read_text()
    if ENV:
        inj="import os\n"+"".join(f"os.environ.setdefault({k!r},{v!r})\n" for k,v in ENV.items())
        code=inj+code
    (KERNEL_DIR/"train.py").write_text(code)
    meta={"id":f"{u}/{KERNEL_SLUG}","title":KERNEL_SLUG,"code_file":"train.py","language":"python",
          "kernel_type":"script","is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4",
          "enable_internet":True,"dataset_sources":[f"{u}/{DATASET_SLUG}"],
          "competition_sources":[],"kernel_sources":[]}
    (KERNEL_DIR/"kernel-metadata.json").write_text(json.dumps(meta,indent=2))
    a.kernels_push(str(KERNEL_DIR))
    print("pushed:",meta["id"],"| env:",ENV or "(defaults)")
    print("view: https://www.kaggle.com/code/%s/%s"%(u,KERNEL_SLUG))

def do_status():
    a=api(); u=user(a)
    try:
        r=a.kernels_status(f"{u}/{KERNEL_SLUG}")
    except Exception as e:
        print("status: PENDING (session not registered yet)"); return "PENDING"
    st=getattr(r,"status",r); fm=getattr(r,"failure_message","") or getattr(r,"failureMessage","")
    print("status:",st, ("| "+str(fm)) if fm else ""); return str(st)

def do_fetch():
    a=api(); u=user(a); OUT_DIR.mkdir(parents=True,exist_ok=True)
    a.kernels_output(f"{u}/{KERNEL_SLUG}",path=str(OUT_DIR))
    for p in sorted(OUT_DIR.glob("*")): print(" ",p.name,p.stat().st_size,"bytes")
    m=OUT_DIR/"oof_metrics.json"
    if m.exists(): print("\nOOF METRICS:\n",m.read_text())

if __name__=="__main__":
    {"dataset":do_dataset,"push":do_push,"status":do_status,"fetch":do_fetch}[sys.argv[1] if len(sys.argv)>1 else "status"]()
