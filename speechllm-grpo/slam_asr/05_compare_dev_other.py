"""YZM5110 v1.0.1 | Yeni: Uzun sesler örtüşmesiz 30 saniyelik parçalara ayrılır. dev-other üzerinde eşleştirilmiş pilot değerlendirmesi.
Girdi: dev-other.tar.gz ve iki pilot koşusunun config.json/projector.pt dosyaları.
Çıktı: tam manifest, sabit örneklem, referans/tahmin ve hata dökümleri.
Seçilen sesler Colab yerel diskine, deney sonuçları Drive'a yazılır.
"""
import argparse
import json
import random
import re
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(archive, count, seed, work, out):
    # Birinci geçişte yalnızca transkriptleri oku. Seçimi arşiv sırasından bağımsız yap.
    refs = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.startswith("LibriSpeech/dev-other/") and member.name.endswith(".trans.txt"):
                for line in tar.extractfile(member):
                    uid, text = line.decode().strip().split(maxsplit=1)
                    if uid in refs or not re.fullmatch(r"\d+-\d+-\d+", uid):
                        raise ValueError(f"Geçersiz veya yinelenen kayıt: {uid}")
                    refs[uid] = dict(id=uid, text=text, member=str(Path(member.name).parent / (uid + ".flac")))
    if not 1 <= count <= len(refs):
        raise ValueError(f"Örneklem {count}, bulunan kayıt {len(refs)}")
    write_jsonl(out / "dev-other.jsonl", [refs[k] for k in sorted(refs)])
    rows = [refs[k] for k in random.Random(seed).sample(sorted(refs), count)]
    write_jsonl(out / "sample.jsonl", rows)
    # İkinci geçişte yalnız seçilen FLAC dosyalarını yerel diske çıkar.
    wanted = {r["member"]: r["id"] for r in rows}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name in wanted and member.isfile():
                with (work / (wanted.pop(member.name) + ".flac")).open("wb") as f:
                    shutil.copyfileobj(tar.extractfile(member), f)
            if not wanted:
                break
    if wanted:
        raise ValueError(f"Eşleşmeyen ses dosyaları: {list(wanted)[:3]}")
    return rows


def errors(reference, prediction):
    # Aynı normalizasyon iki kola da uygulanır. Apostroflar korunur.
    def words(text):
        return re.findall(r"[A-Z0-9]+(?:'[A-Z0-9]+)*", text.upper().replace("’", "'"))
    ref, hyp = words(reference), words(prediction)
    # Hücre: toplam düzenleme, değiştirme, silme, ekleme.
    # Eşit maliyette öncelik: değiştirme, silme, ekleme.
    prev = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, word in enumerate(ref, 1):
        curr = [(i, 0, i, 0)]
        for j, token in enumerate(hyp, 1):
            if word == token:
                curr.append(prev[j-1])
                continue
            a, b, c = prev[j-1], prev[j], curr[-1]
            choices = [(a[0]+1,a[1]+1,a[2],a[3]), (b[0]+1,b[1],b[2]+1,b[3]),
                       (c[0]+1,c[1],c[2],c[3]+1)]
            curr.append(min(choices, key=lambda x: x[0]))
        prev = curr
    _, sub, delete, ins = prev[-1]
    return dict(N=len(ref), S=sub, D=delete, I=ins)


def audio_chunks(wave, rate, uid):
    if rate != 16000 or wave.ndim != 1 or not len(wave):
        raise ValueError(f"{uid}: rate={rate}, shape={wave.shape}; 16 kHz mono ses bekleniyor")
    window = 30 * rate
    return (wave[start:start + window] for start in range(0, len(wave), window))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/slam_asr"))
    parser.add_argument("--runs", nargs=2, default=["A_pilot_20261112T073451453968Z","A_pilot_20261112T073702175976Z"])
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()
    if min(args.samples,args.num_beams,args.max_new_tokens)<1:
        parser.error("Örnek, beam ve token sayıları pozitif olmalı.")
    import torch
    import soundfile as sf
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel
    if not torch.cuda.is_available():
        raise RuntimeError("GPU oturumu gerekiyor.")
    runs = [args.root / "runs" / name for name in args.runs]
    configs = [json.loads((r / "config.json").read_text()) for r in runs]
    for r in runs:
        if not (r / "projector.pt").is_file():
            raise FileNotFoundError(r / "projector.pt")
    cfg = configs[0]
    for key in ("whisper","vicuna","whisper_revision","vicuna_revision","dtype"):
        if configs[1][key] != cfg[key]:
            raise ValueError(f"Koşular arasında {key} farklı")
    out = args.root / "runs" / datetime.now(timezone.utc).strftime("dev_compare_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    print("Çıktı dizini:",out,flush=True)
    dtype = getattr(torch, cfg["dtype"].split(".")[-1])
    with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
        work = Path(tmp)
        rows = prepare(args.root/"data/archives/dev-other.tar.gz", args.samples,args.seed,work,out)
        feature = WhisperFeatureExtractor.from_pretrained(cfg["whisper"], revision=cfg["whisper_revision"])
        encoder = WhisperModel.from_pretrained(cfg["whisper"], revision=cfg["whisper_revision"]).encoder.cuda().to(dtype).eval()
        cached = []
        for row in rows:
            wave, rate = sf.read(work/(row["id"]+".flac"),dtype="float32")
            chunks = []
            for chunk in audio_chunks(wave, rate, row["id"]):
                x = feature(chunk,sampling_rate=rate,return_tensors="pt").input_features.cuda().to(dtype)
                frames = encoder(x).last_hidden_state[0]
                t,h = frames.shape
                chunks.append(frames[:t//5*5].reshape(-1,5*h).cpu())
            if len(chunks)>1:
                print(f"{row['id']}: {len(wave)/rate:.2f} saniye, {len(chunks)} parça",flush=True)
            cached.append(chunks)
        del encoder
        tok = AutoTokenizer.from_pretrained(cfg["vicuna"],revision=cfg["vicuna_revision"])
        llm = AutoModelForCausalLM.from_pretrained(cfg["vicuna"],revision=cfg["vicuna_revision"],torch_dtype=dtype).cuda().eval()
        emb = llm.get_input_embeddings()
        def embed(ids):
            return emb(torch.tensor(ids,device="cuda"))
        # Aynı şablon, aynı sesler, aynı çözücü. Referans yalnızca puanlamada kullanılır.
        prefix = embed([tok.bos_token_id]+tok.encode("USER: ",add_special_tokens=False))
        prompt = embed(tok.encode(" Transcribe speech to text. ASSISTANT:",add_special_tokens=False))
        summary = []
        for run, config in zip(runs,configs):
            projector = nn.Sequential(nn.Linear(5*h,2048),nn.ReLU(),nn.Linear(2048,emb.embedding_dim)).cuda().eval()
            projector.load_state_dict(torch.load(run/"projector.pt",map_location="cpu",weights_only=True))
            totals = dict(N=0,S=0,D=0,I=0)
            with (out/(run.name+".jsonl")).open("w") as log:
                for i,(row,sound) in enumerate(zip(rows,cached),1):
                    # Her pencere pilotun giriş biçimini korur. WER tüm kayıtta bir kez hesaplanır.
                    parts = []
                    for chunk in sound:
                        audio = projector(chunk.cuda().float()).to(dtype)
                        ctx = torch.cat([prefix,audio,prompt]).unsqueeze(0)
                        ids = llm.generate(inputs_embeds=ctx, attention_mask=torch.ones(ctx.shape[:2],dtype=torch.long,device="cuda"),
                                           do_sample=False,num_beams=args.num_beams,max_new_tokens=args.max_new_tokens,pad_token_id=tok.eos_token_id)
                        parts.append(tok.decode(ids[0],skip_special_tokens=True).strip())
                    hyp = " ".join(parts).strip()
                    counts = errors(row["text"],hyp)
                    for key in totals:
                        totals[key] += counts[key]
                    log.write(json.dumps(dict(id=row["id"],chunks=len(sound),reference=row["text"],prediction=hyp,**counts))+"\n")
                    if i%8==0:
                        print(run.name,i,"/",len(rows),flush=True)
            wer = 100*(totals["S"]+totals["D"]+totals["I"])/totals["N"]
            result = dict(run=run.name,lr=config["lr"],wer_percent=wer,**totals)
            summary.append(result)
            print(result,flush=True)
        (out/"summary.json").write_text(json.dumps(dict(version="1.0.1",samples=args.samples,seed=args.seed,
            num_beams=args.num_beams,max_new_tokens=args.max_new_tokens,results=summary),indent=2))


if __name__ == "__main__":
    main()
