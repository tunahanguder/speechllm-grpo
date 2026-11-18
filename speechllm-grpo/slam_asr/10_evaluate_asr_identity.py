"""YZM5110 • v1.1.0 • LibriSpeech üzerinde A/B/C checkpoint değerlendirmesi.

Her kayıt için referans, tahmin ve kelime hata sayılarını saklar. Tamamlanan
model/checkpoint/bölüm kombinasyonu yeniden çalıştırıldığında diskten okunur.
Çalıştırma sırasında 05_compare_dev_other.py bu dosyanın yanında olmalı.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", action="version", version="1.1.0")
    p.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/slam_asr"))
    p.add_argument("--runs", nargs="+", required=True, help="runs içindeki tamamlanmış eğitim klasörleri")
    p.add_argument("--checkpoints", nargs="+", choices=["best","last"], required=True)
    p.add_argument("--splits", nargs="+", choices=["dev-other","test-clean","test-other"], required=True)
    p.add_argument("--beams", type=int, required=True, help="1 greedy; 4 beam search")
    p.add_argument("--max-new-tokens", type=int, required=True)
    p.add_argument("--output", required=True, help="runs altındaki değerlendirme klasörü")
    p.add_argument("--loss", action="store_true", help="Referans token başına teacher-forced çapraz entropi de hesapla")
    a = p.parse_args()
    if min(a.beams,a.max_new_tokens)<1:
        p.error("Beam ve token sınırı pozitif olmalı.")
    for name in [a.output,*a.runs]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+",name):
            p.error("Klasör adı tek bileşen olmalı.")
    if a.output in a.runs:
        p.error("Değerlendirme çıktısı için ayrı klasör seçin.")
    return a


def prepare(archive, split, work):
    # Üç bölümde de aynı kurallar: sadece ilgili dizindeki FLAC ve transkriptler.
    # Arşivi bir kez sırayla okur, her kayıt kimliğine tek referans eşleriz.
    refs, sounds = {}, set()
    with tarfile.open(archive,"r|gz") as tar:
        for item in tar:
            if not item.isfile() or not item.name.startswith("LibriSpeech/"+split+"/"):
                continue
            name = Path(item.name)
            if name.suffix == ".flac":
                uid = name.stem
                if not re.fullmatch(r"\d+-\d+-\d+",uid) or uid in sounds:
                    raise ValueError(f"Geçersiz veya çift ses: {uid}")
                with tar.extractfile(item) as src, (work/(uid+".flac")).open("wb") as dst:
                    shutil.copyfileobj(src,dst)
                sounds.add(uid)
            elif name.name.endswith(".trans.txt"):
                with tar.extractfile(item) as src:
                    for line in src:
                        uid, ref = line.decode().strip().split(maxsplit=1)
                        if uid in refs:
                            raise ValueError(f"Çift transkript: {uid}")
                        refs[uid] = ref
    if not sounds or sounds != refs.keys():
        raise ValueError(f"{split}: FLAC ve referanslar eksik veya eşleşmiyor.")
    return refs


def write_json(path, data):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    tmp.replace(path)


def evaluate(a, run, split, refs, work, out, dev):
    import torch
    import soundfile as sf
    from torch import nn
    from transformers import AutoTokenizer, AutoModelForCausalLM, WhisperModel, WhisperFeatureExtractor

    base=a.root/"runs"/run
    cfg=json.loads((base/"config.json").read_text())
    run_cfg = cfg
    is_adapter = "b_config" in run_cfg
    if is_adapter:
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        cfg = run_cfg["b_config"]
        projector = a.root / "runs" / run_cfg["b_run"] / "best_projector.pt"
        if hashlib.sha256(projector.read_bytes()).hexdigest() != run_cfg["b_sha256"]:
            raise ValueError("B-best projector eğitimde kullanılan ağırlıklarla eşleşmiyor.")
        adapter_cfg = LoraConfig.from_pretrained(str(base))
    is_qwen="qwen" in cfg
    key="qwen" if is_qwen else "vicuna"
    dtype=getattr(torch,cfg["dtype"].split(".")[-1])
    model_cls=AutoModelForCausalLM
    if is_qwen:
        from transformers import Qwen3_5ForCausalLM
        model_cls=Qwen3_5ForCausalLM
    pending=[]
    results=[]
    for kind in a.checkpoints:
        # Eğitim betiklerinde last.pt optimizer durumunu da taşır. Son projector'ın
        # sade state_dict'i projector.pt dosyasındadır.
        ck=base/("best_projector.pt" if kind=="best" else "projector.pt")
        if is_adapter:
            ck = base / ("best_adapter.pt" if kind == "best" else "adapter.pt")
        sha=hashlib.sha256(ck.read_bytes()).hexdigest()
        folder=out/run/kind/split
        folder.mkdir(parents=True,exist_ok=True)
        signature=dict(run=run,checkpoint=kind,sha256=sha,split=split,beams=a.beams,
                       max_new_tokens=a.max_new_tokens,loss_enabled=a.loss,config=cfg,version="1.0.0")
        if is_adapter:
            signature.update(config=run_cfg, adapter_config_sha256=hashlib.sha256((base/"adapter_config.json").read_bytes()).hexdigest(), version="1.1.0")
        summary=folder/"summary.json"
        if summary.exists():
            saved=json.loads(summary.read_text())
            if saved["protocol"]!=signature or not (folder/"predictions.jsonl").exists():
                raise ValueError(f"{folder}: dosyalar veya protokol farklı; yeni --output seçin.")
            results.append(saved["metrics"])
        else:
            pending.append((kind,ck,folder,signature))
    if not pending:
        return results

    feature=WhisperFeatureExtractor.from_pretrained(cfg["whisper"],revision=cfg["whisper_revision"])
    whisper=WhisperModel.from_pretrained(cfg["whisper"],revision=cfg["whisper_revision"])
    width=whisper.config.d_model
    encoder=whisper.encoder.to("cuda",dtype).eval()
    del whisper
    tok=AutoTokenizer.from_pretrained(cfg[key],revision=cfg[key+"_revision"])
    lm,info=model_cls.from_pretrained(cfg[key],revision=cfg[key+"_revision"],torch_dtype=dtype,
                                       output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"{run}: dil ağırlıkları tam yüklenmedi.")
    lm=lm.cuda().eval()
    emb=lm.get_input_embeddings()
    proj=nn.Sequential(nn.Linear(width*5,2048),nn.Identity(),nn.Linear(2048,emb.embedding_dim)).cuda().eval()

    if is_adapter:
        # Tüm C checkpoint'leri eğitimdeki aynı donuk B-best projector'ını kullanır.
        proj.load_state_dict(torch.load(projector, map_location="cpu", weights_only=True))
        lm = get_peft_model(lm, adapter_cfg).eval()

    def ids(text):
        return tok.encode(text,add_special_tokens=False)

    def embed(seq):
        return emb(torch.tensor(seq,device="cuda"))

    with torch.inference_mode():
        prefix=embed(([] if tok.bos_token_id is None else [tok.bos_token_id])+ids("USER: "))
        prompt=embed(ids(" Transcribe speech to text. ASSISTANT:"))
        for kind,ck,folder,signature in pending:
            weights = torch.load(ck, map_location="cpu", weights_only=True)
            if is_adapter:
                # Eksik bir LoRA tensörü önceki checkpoint'ten kalmasın.
                if weights.keys() != get_peft_model_state_dict(lm).keys():
                    raise ValueError(f"LoRA anahtarları eşleşmiyor: {ck}")
                set_peft_model_state_dict(lm, weights)
            else:
                proj.load_state_dict(weights)
            totals=dict(N=0,S=0,D=0,I=0)
            tokens=long_count=cap_hits=0
            nll=0.
            tmp=folder/"predictions.partial"
            with tmp.open("w",encoding="utf-8") as writer:
                for j,(uid,ref) in enumerate(sorted(refs.items()),1):
                    w,rate=sf.read(work/(uid+".flac"),dtype="float32")
                    parts,sounds=[],[]
                    for chunk in dev.audio_chunks(w,rate,uid):
                        x=feature(chunk,sampling_rate=rate,return_tensors="pt").input_features
                        h=encoder(x.to("cuda",dtype)).last_hidden_state[0]
                        sound=proj(h[:len(h)//5*5].reshape(-1,width*5).float()).to(dtype)
                        sounds.append(sound)
                        ctx=torch.cat((prefix,sound,prompt))[None]
                        mask=torch.ones(ctx.shape[:2],device="cuda",dtype=torch.long)
                        pred=lm.generate(inputs_embeds=ctx,attention_mask=mask,do_sample=False,
                                         num_beams=a.beams,max_new_tokens=a.max_new_tokens,
                                         pad_token_id=tok.eos_token_id)[0]
                        cap_hits+=int(len(pred)==a.max_new_tokens)
                        parts.append(tok.decode(pred,skip_special_tokens=True))
                    hyp=" ".join(parts)
                    c=dev.errors(ref,hyp)
                    for k in totals:
                        totals[k]+=c[k]
                    long_count+=int(len(parts)>1)
                    row=dict(id=uid,reference=ref,hypothesis=hyp,chunks=len(parts),**c)
                    if a.loss:
                        # Uzun kayıtta parça düzeyinde doğru metin hizası yoktur.
                        # Tüm ses temsillerini sırayla bağla; tam referansı bir kez puanla.
                        # WER'de ise 05'teki gibi her parça bağımsız üretilir.
                        ctx=torch.cat([prefix,*sounds,prompt])
                        target=ids(" "+ref)+[tok.eos_token_id]
                        inp=torch.cat((ctx,embed(target)))[None]
                        labels=torch.tensor([-100]*len(ctx)+target,device="cuda")[None]
                        loss=lm(inputs_embeds=inp,labels=labels,use_cache=False).loss.item()
                        row.update(target_tokens=len(target),mean_token_loss=loss)
                        nll+=loss*len(target)
                        tokens+=len(target)
                    writer.write(json.dumps(row,ensure_ascii=False)+"\n")
                    if j%100==0:
                        print(run,kind,split,j,"/",len(refs),flush=True)
            tmp.replace(folder/"predictions.jsonl")
            m=dict(run=run,checkpoint=kind,split=split,records=len(refs),**totals,
                   wer_percent=100*(totals["S"]+totals["D"]+totals["I"])/totals["N"],
                   multi_chunk_records=long_count,chunks_at_token_limit=cap_hits,
                   mean_token_loss=nll/tokens if tokens else None)
            write_json(folder/"summary.json",dict(protocol=signature,metrics=m))
            print(m,flush=True)
            results.append(m)
    return results


def main():
    a=arguments()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU oturumu gerekiyor.")
    spec=importlib.util.spec_from_file_location("dev",Path(__file__).with_name("05_compare_dev_other.py"))
    dev=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev)
    out=a.root/"runs"/a.output
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    for split in a.splits:
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            refs=prepare(a.root/"data/archives"/(split+".tar.gz"),split,work)
            for run in a.runs:
                results.extend(evaluate(a,run,split,refs,work,out,dev))
    path=out/"comparison.csv"
    with path.with_suffix(".tmp").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    path.with_suffix(".tmp").replace(path)
    print(path)


if __name__=="__main__":
    main()
