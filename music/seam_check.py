import sys,soundfile as sf,numpy as np
z,sr=sf.read(sys.argv[1]); P=float(sys.argv[2]); J=len(z); t=np.concatenate([z,z]); m=t.mean(1); b=int(P*sr)
print("samples",J,"sec",J/sr,"join step",np.abs(t[J]-t[J-1]).max(),"max step elsewhere",np.abs(np.diff(t,axis=0)).max(),"p99.9",np.percentile(np.abs(np.diff(t,axis=0)),99.9))
r=[np.sqrt((m[J+i*b:J+(i+1)*b]**2).mean()) for i in range(-8,8)]
print("beat RMS around join (-8..+7 beats):",np.round(r,3)); ov=np.array([np.sqrt((m[i:i+b]**2).mean()) for i in range(0,J-b,b)])
print("beat RMS elsewhere median",ov.mean().round(3),"p5-p95",np.percentile(ov,5).round(3),np.percentile(ov,95).round(3),"; adjacent-beat ratio at join",round(max(r[7]/r[8],r[8]/r[7]),2),"vs elsewhere p90",np.percentile(np.maximum(ov[1:]/ov[:-1],ov[:-1]/ov[1:]),90).round(2))
