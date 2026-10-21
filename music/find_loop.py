#!/usr/bin/env python3
"""Search a raw generation for the best loop start (content-similarity at the join), then write the loop.
usage: find_loop.py RAW.wav P T0 L_BEATS OUT.wav [--old OLD1.wav,OLD2.wav] [--xfade 2] [--step 2] [--min-start 3] [--max-end 94] [--top 8]
Candidates: start = T0 + k*STEP*P. Loop = raw[start:start+L] with first XFADE beats replaced by an equal-power blend of the head with
raw[start+L : start+L+X] (so the end flows into the head). Join score = mean over features of
|feature(2-bar window after join) - feature(2-bar window before join)| / median |same distance between adjacent windows elsewhere in the raw|.
(1.0 = as different as two neighbouring places in the music; lower = smoother.) Features: chroma, log-mel, centroid, log-RMS, onset density."""
import sys, argparse, numpy as np, soundfile as sf, librosa
ap=argparse.ArgumentParser(); ap.add_argument("raw"); ap.add_argument("P",type=float); ap.add_argument("t0",type=float)
ap.add_argument("L",type=int); ap.add_argument("out"); ap.add_argument("--old",default=""); ap.add_argument("--xfade",type=float,default=2)
ap.add_argument("--step",type=float,default=2); ap.add_argument("--min-start",type=float,default=3); ap.add_argument("--max-end",type=float,default=94)
ap.add_argument("--top",type=int,default=8); ap.add_argument("--bars",type=float,default=2)
a=ap.parse_args(); SR=22050; HOP=512
x48,sr48=sf.read(a.raw,dtype="float64",always_2d=True); m=librosa.resample(x48.mean(1),orig_sr=sr48,target_sr=SR)
W=int(round(a.bars*4*a.P*SR)); P=a.P
def feats(y):
    S=np.abs(librosa.stft(y,n_fft=2048,hop_length=HOP))
    ch=librosa.feature.chroma_stft(S=S**2,sr=SR); ch=ch.mean(1); ch/=ch.sum()+1e-9
    mel=np.log(librosa.feature.melspectrogram(S=S**2,sr=SR,n_mels=40)+1e-6).mean(1)
    cen=librosa.feature.spectral_centroid(S=S,sr=SR).mean()
    rms=np.log(np.sqrt((y**2).mean())+1e-5)
    on=librosa.onset.onset_strength(S=librosa.power_to_db(S**2),sr=SR).mean()
    return dict(chroma=ch,mel=mel,cen=cen,rms=rms,onset=on)
def dist(f,g):
    return dict(chroma=np.linalg.norm(f["chroma"]-g["chroma"]),mel=np.linalg.norm(f["mel"]-g["mel"])/np.sqrt(40),
        cen=abs(f["cen"]-g["cen"]),rms=abs(f["rms"]-g["rms"]),onset=abs(f["onset"]-g["onset"]))
# typical adjacent-window distances elsewhere
t=[];s=int(a.min_start*SR)
while s+2*W<=len(m) and s+2*W<=a.max_end*SR: t.append(dist(feats(m[s:s+W]),feats(m[s+W:s+2*W]))); s+=W//2
keys=list(t[0]); typ={k:np.median([d[k] for d in t]) for k in keys}
def score_region(tail,head):
    d=dist(feats(tail),feats(head)); n={k:d[k]/typ[k] for k in keys}; return float(np.mean(list(n.values()))),n
def beat_rms_ratio(tail,head):
    b=int(P*SR); r1=np.sqrt((tail[-b:]**2).mean())+1e-6; r2=np.sqrt((head[:b]**2).mean())+1e-6; return max(r1/r2,r2/r1)
def build(x,start_s,L,X):  # x: (n,ch) any rate; start_s in samples of x
    t_=np.linspace(0,np.pi/2,X,endpoint=False)[:,None]; y=x[start_s:start_s+L].copy()
    y[:X]=x[start_s:start_s+X]*np.sin(t_)+x[start_s+L:start_s+L+X]*np.cos(t_); return y
Lsec=a.L*P; X22=int(round(a.xfade*P*SR)); L22=int(round(a.L*P*SR)); rows=[]
k=0
while True:
    st=a.t0+k*a.step*P; k+=1
    if st<a.min_start: continue
    if st+Lsec+a.xfade*P>a.max_end: break
    s22=int(round(st*SR)); y=build(m[:,None],s22,L22,X22)[:,0]
    tail,head=y[-W:],y[:W]; sc,n=score_region(tail,head); rr=beat_rms_ratio(y,y)
    tailbeats=np.sqrt((y[-int(2*P*SR):]**2).mean())/ (np.sqrt((y[-W:-int(2*P*SR)]**2).mean())+1e-6)
    ok = rr<=1.5 and tailbeats>0.5
    rows.append((sc,st,rr,tailbeats,ok,n))
rows.sort(key=lambda r:(not r[4],r[0]))
print("typical adjacent distances:",{k:round(float(v),4) for k,v in typ.items()})
print("rank  start_s  score  rms1beat_ratio  endDecayRatio  ok   (chroma mel cen rms onset normalised)")
for i,(sc,st,rr,tb,ok,n) in enumerate(rows[:a.top]): print(f"{i+1:3d} {st:8.3f} {sc:6.2f} {rr:6.2f} {tb:6.2f} {ok!s:5} "+" ".join(f"{n[k]:.2f}" for k in keys))
print("n candidates",len(rows),"; median score",round(float(np.median([r[0] for r in rows])),2),"best",round(rows[0][0],2))
for p in [q for q in a.old.split(",") if q]:   # score old loops the same way (join = wrap-around)
    o,sro=sf.read(p,dtype="float64",always_2d=True); o=librosa.resample(o.mean(1),orig_sr=sro,target_sr=SR); sc,n=score_region(o[-W:],o[:W])
    print("OLD",p,"score",round(sc,2),{k:round(v,2) for k,v in n.items()},"rms1beat ratio",round(beat_rms_ratio(o,o),2))
best=rows[0]; st=best[1]; L=int(round(a.L*P*sr48)); X=int(round(a.xfade*P*sr48)); s=int(round(st*sr48))
y=build(x48,s,L,X); y*=0.891/np.abs(y).max(); sf.write(a.out,y,sr48,subtype="PCM_16")
print("WROTE",a.out,"start",round(st,3),"samples",len(y),"sec",len(y)/sr48,"score",round(best[0],2))
