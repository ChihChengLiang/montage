#!/usr/bin/env python3
"""Synthesise a 5.000 s opener sting on a 144 bpm grid (12 beats x 20000 samples @48k). Deterministic (seeded)."""
import numpy as np, soundfile as sf, json
from scipy.signal import butter, sosfilt
sr=48000; B=20000; N=12*B; rng=np.random.default_rng(5)
t_all=np.arange(N)/sr
L=np.zeros(N); R=np.zeros(N)
def add(sig,at,gain=1.0,pan=0.0):
    a=int(round(at)); e=min(N,a+len(sig)); 
    if a>=N: return
    s=sig[:e-a]*gain; L[a:e]+=s*np.cos((pan+1)*np.pi/4); R[a:e]+=s*np.sin((pan+1)*np.pi/4)
def taiko(dur=0.32,f0=135,f1=46,dec=7.5):
    t=np.arange(int(dur*sr))/sr; f=f1+(f0-f1)*np.exp(-t*28); ph=2*np.pi*np.cumsum(f)/sr
    body=np.sin(ph)*np.exp(-t*dec)
    n=sosfilt(butter(2,[400,3000],'bp',fs=sr,output='sos'),rng.standard_normal(len(t)))*np.exp(-t*60)*0.35  # skin slap
    return body+n+0.3*np.sin(2*ph)*np.exp(-t*14)
def tick(f=2200,dec=140,dur=0.06):
    t=np.arange(int(dur*sr))/sr; n=sosfilt(butter(2,5000,'hp',fs=sr,output='sos'),rng.standard_normal(len(t)))
    return (0.6*np.sin(2*np.pi*f*t)+0.7*n)*np.exp(-t*dec)
tk=taiko(); sub_hit=lambda g:taiko(0.5,70,38,4.5)*g
# pulse: quarter notes, then 8ths, 16ths roll
hits=[]  # (beat_pos, gain)
for b in range(0,4): hits.append((b,0.75+0.05*b))
for b in range(4,8):
    hits += [(b,0.85),(b+0.5,0.5)]
for b in range(8,11):
    hits += [(b+k/4,0.9 if k==0 else 0.55) for k in range(4)]
hits += [(11+k/8,0.65+0.35*k/6) for k in range(0,6)]   # 32nd roll up to 11.625 before the hit at 11.75
for pos,g in hits: add(tk,pos*B,g*0.9,0.0)
# ticks accelerate: 8ths (beats0-3) -> 16ths (4-7) -> 32nds (8-10), gain rises
pos=0.0; i=0
while pos<11.5:
    d=0.5 if pos<4 else (0.25 if pos<8 else 0.125)
    g=0.10+0.30*(pos/11.5); add(tick(2200 if i%2==0 else 3100),pos*B,g,pan=-0.5 if i%2==0 else 0.5); pos+=d; i+=1
# sub bass drone D1 (36.7Hz) pumping with beats and rising in level
f=36.71*(1+0.03*t_all/5); drone=np.sin(2*np.pi*np.cumsum(f)/sr)
pump=0.55+0.45*np.abs(np.cos(np.pi*t_all/(B/sr)))**3   # ducks between beats
env=0.10+0.30*(t_all/5.0)**1.5
sub=drone*pump*env; L+=sub; R+=sub
# riser over last 2 beats (beat 10 -> 11.75 hit)
a=int(10*B); b=int(11.75*B); n=b-a; tt=np.arange(n)/sr; u=np.linspace(0,1,n)
noise=rng.standard_normal((2,n)); out=np.zeros((2,n))
# time-varying band-pass via block filtering with rising centre
blk=2400
for s in range(0,n,blk):
    e=min(n,s+blk); c=300*(40**(s/n)); lo=max(60,c*0.6); hi=min(22000,c*1.8)
    sos=butter(2,[lo,hi],'bp',fs=sr,output='sos')
    for ch in range(2): out[ch,s:e]=sosfilt(sos,noise[ch,s-0 if False else s:e])
rise=u**2.2; out*=rise*0.9
fr=500*(12**(u)); tone=np.sin(2*np.pi*np.cumsum(fr)/sr)*rise*0.12
L[a:b]+=out[0]/np.abs(out).max()*0.55+tone; R[a:b]+=out[1]/np.abs(out).max()*0.55+tone
# impact at beat 11.75: sub boom + taiko + noise burst
h0=int(round(11.75*B)); hl=N-h0
t=np.arange(hl)/sr
boom=np.sin(2*np.pi*np.cumsum(30+110*np.exp(-t*30))/sr)*np.exp(-t*6)*1.3
burst=sosfilt(butter(2,[150,9000],'bp',fs=sr,output='sos'),rng.standard_normal(hl))*np.exp(-t*30)*0.9
imp=boom+burst+taiko(hl/sr,150,40,5)[:hl]*1.2; imp[:24]*=np.linspace(0,1,24)
# short dip right before the hit (silence-of-anticipation 8 ms) to make the hit decisive
dip=int(0.008*sr); L[h0-dip:h0]*=np.linspace(1,0.3,dip); R[h0-dip:h0]*=np.linspace(1,0.3,dip)
L[h0:]+=imp; R[h0:]+=imp
# master: soft clip, edges, normalise to -1 dBFS
y=np.stack([L,R],1); y=np.tanh(1.4*y)
y[:48]*=np.linspace(0,1,48)[:,None]
fo=int(0.005*sr); y[-fo:]*=np.linspace(1,0,fo)[:,None]
y*=10**(-1/20)/np.abs(y).max()
sf.write("eyelock_sting.wav",y,sr,subtype="PCM_16")
z,_=sf.read("eyelock_sting.wav"); assert z.shape==(240000,2)
m=z.mean(1); rms=[float(np.sqrt((m[i*B:(i+1)*B]**2).mean())) for i in range(12)]
info=dict(bpm=144,beat_seconds=B/sr,first_beat_offset_s=0.0,length_s=5.0,hit_time_s=h0/sr,beats=12,rms_per_beat=[round(r,4) for r in rms],peak_dbfs=-1.0)
json.dump(info,open("sting_info.json","w"),indent=1); print(info)
