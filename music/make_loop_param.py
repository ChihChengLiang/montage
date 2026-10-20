# usage: make_loop_param.py RAW.wav OUT.wav BEAT_PERIOD_SEC BEATS [XFADE_BEATS=2] [START_SEC=0]
# length = round(BEATS*period*48000) samples; first XFADE_BEATS replaced by equal-power blend of head with audio after loop end.
import sys, numpy as np, soundfile as sf
raw,out,P,N=sys.argv[1],sys.argv[2],float(sys.argv[3]),int(sys.argv[4])
xb=float(sys.argv[5]) if len(sys.argv)>5 else 2; st=float(sys.argv[6]) if len(sys.argv)>6 else 0
x,sr=sf.read(raw,dtype="float64",always_2d=True); x=x[int(round(st*sr)):]
L=int(round(N*P*sr)); X=int(round(xb*P*sr)); assert len(x)>=L+X, (len(x),L+X)
t=np.linspace(0,np.pi/2,X,endpoint=False)[:,None]
y=x[:L].copy(); y[:X]=x[:X]*np.sin(t)+x[L:L+X]*np.cos(t)
y*=0.891/np.abs(y).max(); sf.write(out,y,sr,subtype="PCM_16")
z,_=sf.read(out,dtype="float64",always_2d=True); d=np.abs(np.diff(np.concatenate([z,z]),axis=0)).max(1)
print("samples",len(z),"sec",len(z)/sr,"join step",d[len(z)-1],"max other",np.delete(d,len(z)-1).max(),"p99.9",np.percentile(d,99.9))
