import sys, pathlib, tempfile; sys.path.insert(0,"src"); sys.path.insert(0,"tests")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import test_guided_fix as T
class F:
    def mktemp(self,n): p=pathlib.Path(tempfile.mkdtemp())/n; p.mkdir(); return p
proj=T.project.__wrapped__(F())
s=T._session(proj); r=T._wrong_row(s); rows=np.asarray(s.T["rows"],float); x0=s.T["x"].copy()
img=s.img if hasattr(s,"img") else None
from PIL import Image; im=np.array(Image.open(proj/"example_data/SYN_1/syn.tif"))
import time; t0=time.perf_counter(); out=s.node_snap(r,float(T.xa(r)),radius_px=80); out["ms"]=(time.perf_counter()-t0)*1e3; x1=s.T["x"]
fig,ax=plt.subplots(1,2,figsize=(8,7),sharey=True)
for a,x,t in [(ax[0],x0,"before: auto trace on distractor"),(ax[1],x1,"after: node + local ink snap")]:
    a.imshow(im,cmap="gray",aspect="auto"); a.plot(x,rows,"r",lw=1.5); a.set_title(t,fontsize=9)
    a.set_ylim(r+500,r-500); a.set_xlim(0,T.W)
ax[1].add_patch(plt.Circle((T.xa(r),r),80,fill=False,color="c",lw=1.5)); ax[1].plot(T.xa(r),r,"co")
fig.suptitle(f"Manual node editor (D tool): {out.get('ms',0):.0f} ms, no model rerun",fontsize=10)
fig.savefig(sys.argv[1] if len(sys.argv)>1 else "node_editor_demo.png",dpi=110,bbox_inches="tight"); print(out.get("ms"), out.get("sheet_rows"))
