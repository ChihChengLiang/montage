import librosa, numpy as np, soundfile as sf
y48, sr48 = sf.read("music/loop.wav", always_2d=True); N=len(y48)
y, sr = librosa.load("music/loop.wav", sr=22050, mono=True); H=64
o = librosa.onset.onset_strength(y=y, sr=sr, hop_length=H); fps=sr/H; o=(o-o.mean())/o.std(); T=len(y)/sr
def sc(P, off):
    t=np.arange(off,T,P); i=np.round(t*fps).astype(int); i=i[i<len(o)]; return o[i].mean()
for P,name in ((0.8,"75bpm beat"),(0.4,"eighth")):
    offs=np.arange(0,P,0.001); s=[sc(P,x) for x in offs]; j=int(np.argmax(s)); print(name,"best offset",round(offs[j],3),"score",round(s[j],2),"vs mean",round(np.mean(s),2))
offs=np.arange(0,0.4,0.001); off=offs[int(np.argmax([sc(0.4,x) for x in offs]))]
if off>0.2: off-=0.4
n=int(round(off*sr48)); print("rolling by", n, "samples =", n/sr48*1000, "ms")
out=np.roll(y48,-n,axis=0); sf.write("music/loop_aligned.wav", out, sr48, subtype="PCM_16")
y2,_=librosa.load("music/loop_aligned.wav", sr=22050, mono=True)
o2=librosa.onset.onset_strength(y=y2, sr=sr, hop_length=H); o2=(o2-o2.mean())/o2.std()
offs=np.arange(0,0.4,0.001); s=[ (lambda t:(o2[np.round(t*fps).astype(int)[np.round(t*fps).astype(int)<len(o2)]].mean()))(np.arange(x,T,0.4)) for x in offs]
print("after: best eighth offset", round(offs[int(np.argmax(s))],3))
