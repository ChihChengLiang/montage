# usage: gen_gould.py DURATION BPM SEED1 [SEED2 ...]  -> out_gould/seed<S>.wav
import os, sys, time, shutil
REPO="/Users/liangcc/projects/repos/montage/music/repo"
sys.path.insert(0, REPO); os.chdir(REPO); os.environ["TOKENIZERS_PARALLELISM"]="false"
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
DUR=float(sys.argv[1]); BPM=int(sys.argv[2]); SEEDS=[int(s) for s in sys.argv[3:]]
OUT="/Users/liangcc/projects/repos/montage/music/out_gould"; os.makedirs(OUT,exist_ok=True)
CAPTION=("gentle airy instrumental, morning garden, natural history, soft acoustic texture, light plucked harp and kalimba, "
 "occasional glockenspiel touches, soft warm pads, warm low strings, very sparse minimal melody, steady even unobtrusive energy, "
 "no drums, no vocals, no build-up, calm background music bed")
h=AceStepHandler()
msg,ok=h.initialize_service(project_root=REPO,config_path="acestep-v15-turbo",device="auto",offload_to_cpu=False)
if not ok: sys.exit(msg)
for s in SEEDS:
    p=GenerationParams(task_type="text2music",thinking=False,caption=CAPTION,lyrics="[Instrumental]",instrumental=True,
      bpm=BPM,keyscale="G Major",timesignature="4",vocal_language="unknown",duration=DUR,inference_steps=8,guidance_scale=1.0,seed=s)
    t=time.time(); r=generate_music(h,None,params=p,config=GenerationConfig(batch_size=1,audio_format="wav"),save_dir=OUT+"/tmp")
    print("seed",s,r.success,f"{time.time()-t:.0f}s",flush=True)
    shutil.copy(r.audios[0]["path"],f"{OUT}/seed{s}.wav")
