# usage: align_param.py IN.wav OUT.wav GRID_PERIOD_SEC   (rolls so best-fit grid of that period lands on t=0)
import sys, librosa, numpy as np, soundfile as sf
inp,outp,P=sys.argv[1],sys.argv[2],float(sys.argv[3])
y48,sr48=sf.read(inp,always_2d=True)
y,sr=librosa.load(inp,sr=22050,mono=True); H=64
def prep(y):
    o=librosa.onset.onset_strength(y=y,sr=sr,hop_length=H); return (o-o.mean())/o.std()
fps=sr/H; T=len(y)/sr
def score(o,off):
    t=np.arange(off,T,P); i=np.round(t*fps).astype(int); i=i[i<len(o)]; return o[i].mean()
o=prep(y); offs=np.arange(0,P,0.001); s=np.array([score(o,x) for x in offs]); off=offs[int(s.argmax())]
print("grid",P,"best offset",round(off,3),"score",round(s.max(),2),"mean",round(s.mean(),2),"std",round(s.std(),2))
if off>P/2: off-=P
n=int(round(off*sr48)); print("rolling by",n,"samples =",n/sr48*1000,"ms")
sf.write(outp,np.roll(y48,-n,axis=0),sr48,subtype="PCM_16")
y2,_=librosa.load(outp,sr=22050,mono=True); o2=prep(y2); s2=np.array([score(o2,x) for x in offs])
print("after: best offset",round(offs[int(s2.argmax())],3),"score at 0:",round(score(o2,0.0),2),"max",round(s2.max(),2))
