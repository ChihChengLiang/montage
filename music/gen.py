import os, sys, time
REPO="/Users/liangcc/projects/repos/montage/music/repo"
sys.path.insert(0, REPO); os.chdir(REPO)
os.environ["TOKENIZERS_PARALLELISM"]="false"
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music
DUR=float(sys.argv[1]) if len(sys.argv)>1 else 75
CAPTION=("calm elegant cinematic instrumental, traditional Taiwanese flavour, "
 "gentle guzheng and erhu melody over soft piano and warm strings, bamboo flute, "
 "slow, spacious, contemplative, museum and history documentary underscore, no vocals, no drums")
t0=time.time()
h=AceStepHandler()
msg,ok=h.initialize_service(project_root=REPO,config_path="acestep-v15-turbo",device="auto",offload_to_cpu=False)
print(msg, ok, f"init {time.time()-t0:.0f}s", flush=True)
if not ok: sys.exit(1)
p=GenerationParams(task_type="text2music",thinking=False,caption=CAPTION,lyrics="[Instrumental]",
  instrumental=True,bpm=68,keyscale="D Major",timesignature="4",vocal_language="unknown",
  duration=DUR,inference_steps=8,guidance_scale=1.0,seed=42)
t1=time.time()
r=h.__class__ and generate_music(h,None,params=p,config=GenerationConfig(batch_size=1,audio_format="wav"),save_dir="/Users/liangcc/projects/repos/montage/music/out")
print("ok",r.success,r.status_message,f"gen {time.time()-t1:.0f}s")
for a in r.audios: print(a.get("path"))
