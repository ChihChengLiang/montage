import numpy as np, soundfile as sf
x,sr=sf.read("raw_100bpm.wav",dtype="float64"); assert sr==48000
L=3686400; X=57600; print("raw",x.shape, len(x)/sr)
fi=np.sin(np.linspace(0,np.pi/2,X,endpoint=False))[:,None]; fo=np.cos(np.linspace(0,np.pi/2,X,endpoint=False))[:,None]
y=x[:L].copy()
y[:X]=x[:X]*fi+x[L:L+X]*fo
pk=np.abs(y).max(); y=y*(0.891/pk)
sf.write("loop.wav",y,sr,subtype="PCM_16")
z,_=sf.read("loop.wav",dtype="float64"); print("loop",z.shape,len(z)/sr)
dbl=np.concatenate([z,z]); 
d=np.abs(np.diff(dbl,axis=0)).max(axis=1)
J=len(z); print("join step",d[J-1],"median step",np.median(d),"p99.9",np.percentile(d,99.9),"max elsewhere",np.delete(d,J-1).max())
# local RMS around join vs elsewhere
w=2400
r=lambda a:np.sqrt((a**2).mean())
print("rms before/after join",r(dbl[J-w:J]),r(dbl[J:J+w]),"overall",r(z))
# energy stability: rms per beat in 8-beat chunks
b=28800; e=[r(z[i:i+b*8]) for i in range(0,L,b*8)]; print("rms per 8 beats min/max",min(e),max(e))
# tempo: onset envelope autocorr
m=z.mean(1); h=480; env=np.sqrt(np.convolve(m**2,np.ones(h)/h,'same'))[::h]; o=np.maximum(np.diff(env),0); o-=o.mean()
ac=np.correlate(o,o,'full')[len(o)-1:]; lags=np.arange(len(ac))*0.01
for bpm in (90,95,100,105,110): 
    l=60/bpm; print(bpm, ac[int(round(l/0.01))])
