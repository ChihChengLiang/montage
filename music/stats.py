import sys,soundfile as sf,numpy as np
x,sr=sf.read(sys.argv[1]); m=x.mean(1); n=int(sys.argv[2]) if len(sys.argv)>2 else 80
e=np.array([np.sqrt((m[i:i+sr*2]**2).mean()) for i in range(0,sr*n,sr*2)]); print("rms 2s windows 0-%d s: mean"%n,e.mean().round(3),"CV",(e.std()/e.mean()).round(3),"min/max",e.min().round(3),e.max().round(3))
print("rms 90-100:",round(float(np.sqrt((m[90*sr:100*sr]**2).mean())),3) if len(m)>=100*sr else "")
