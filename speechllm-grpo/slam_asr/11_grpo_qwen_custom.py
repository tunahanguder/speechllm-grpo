"""YZM5110 • Deney D • v1.1.0
B-best başlangıcından EM ve parçalı WER ödülüyle Qwen LoRA eğitimi.
05 (WER ve dev verisi) ve 08 (eğitim manifesti ve ses hazırlığı) yanında bulunmalı.
Pilot ve tam koşu aynı algoritmayı kullanır. --help için GPU gerekmez.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import time

VERSION = "1.1.0"


def args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/slam_asr"))
    p.add_argument("--cache-dir", type=Path, default=Path("/content/slam_asr_audio"))
    p.add_argument("--b-run", default="B_qwen08_train960", help="Tamamlanan SFT koşusu")
    p.add_argument("--em-weight", type=float, default=0.4, help="WER payı 1 eksi EM payıdır.")
    p.add_argument("--wer-threshold", type=float, default=0.1, help="Ağır cezanın başladığı hata oranı.")
    p.add_argument("--excess-penalty", type=float, default=2.0, help="Eşik üzerindeki hata payının çarpanı.")
    p.add_argument("--run-name", default="D_qwen08_lora_em_wer")
    p.add_argument("--resume", action="store_true")
    for key, default, desc in [
        ("steps", 1000, "Optimizer adımı bütçesi"),
        ("group-size", 4, "Aynı ses için bağımsız örneklenen aday sayısı"),
        ("sounds-per-step", 2, "Adım başına farklı ses; gradyanlar sesler üzerinde ortalanır"),
        ("max-new-tokens", 160, "Örnekleme uzunluğu üst sınırı"),
        ("eval-every", 200, "Aynı dev örnekleminde açgözlü WER ölçüm aralığı"),
        ("save-every", 50, "Devam checkpoint aralığı"),
        ("seed", 42, "Veri sırası ve örnekleme tohumu"),
    ]:
        p.add_argument("--"+key, type=int, default=default, help=desc)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=.04, help="Sabit B referansına KL cezası")
    p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--em-ignore-case", action="store_true", help="EM hesabında harf büyüklüğünü yok say.")
    a = p.parse_args()
    if not (0 <= a.em_weight <= 1 and 0 <= a.wer_threshold <= 1
            and math.isfinite(a.excess_penalty) and a.excess_penalty >= 1):
        p.error("EM payı ve eşik [0,1] aralığında, ceza çarpanı en az 1 olmalı.")
    if min(a.steps,a.sounds_per_step,a.max_new_tokens,a.eval_every,a.save_every)<1 or a.group_size<2:
        p.error("Bütçeler pozitif, grup boyutu en az iki olmalı.")
    if not (math.isfinite(a.lr) and a.lr>0 and math.isfinite(a.beta) and a.beta>=0 and 0<a.clip<1 and a.seed>=0):
        p.error("LR, beta, kırpma veya seed geçersiz.")
    for name in (a.b_run, a.run_name):
        if name in {"", ".", ".."} or Path(name).name != name:
            p.error("Koşu adı tek bir klasör adı olmalı.")
    return a


def sibling(name):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_match(reference, hypothesis, ignore_case):
    reference, hypothesis = reference.strip(), hypothesis.strip()
    if ignore_case:
        reference, hypothesis = reference.casefold(), hypothesis.casefold()
    return reference == hypothesis


def reward(reference, hypothesis, word_errors, settings):
    # EM yazıma duyarlı; WER ise değerlendirmedeki kelime normalizasyonunu kullanır.
    exact = float(exact_match(reference, hypothesis, settings.em_ignore_case))
    counts = word_errors(reference, hypothesis)
    if counts["N"] <= 0:
        raise ValueError("WER için referansta en az bir kelime gerekli.")
    wer = (counts["S"] + counts["D"] + counts["I"]) / counts["N"]

    # Örneğin 30 kelimede üç hataya kadar normal ceza uygulanır.
    # Çarpan yalnızca eşiği aşan kısma uygulanır, ilk üç hataya geri dönülmez.
    threshold = settings.wer_threshold
    penalty = min(wer, threshold) + settings.excess_penalty * max(0.0, wer - threshold)
    word_score = max(0.0, 1.0 - penalty)
    return settings.em_weight * exact + (1.0 - settings.em_weight) * word_score


def main():
    a = args()
    import soundfile as sf
    import torch
    from torch import nn
    from transformers import (AutoTokenizer, GenerationConfig, Qwen3_5ForCausalLM,
                              WhisperFeatureExtractor, WhisperModel)
    import peft
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
    import transformers

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 destekli CUDA GPU gerekli.")
    b = a.root/"runs"/a.b_run
    src = b/"best_projector.pt"
    cfg = json.loads((b/"config.json").read_text())
    if cfg["mode"] != "train" or not (b/"projector.pt").exists():
        raise ValueError("Önce B tam eğitimini tamamlayın.")
    B = sibling("08_qwen35_asr.py")
    dev = sibling("05_compare_dev_other.py")
    if cfg["qwen"] != B.QWEN or cfg["whisper"] != B.WHISPER:
        raise ValueError("B model kimlikleri 08 betiği ile uyuşmuyor.")
    out = a.root/"runs"/a.run_name
    last = out/"last.pt"
    if a.resume and not last.is_file():
        raise FileNotFoundError(last)
    if not a.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out}: devam için --resume veya yeni bir koşu adı kullanın.")
    # 08 ile aynı 960 saatlik havuzu ve yerel FLAC dosyalarını kullan.
    a.mode = "train"
    rows, digest = B.train_data(a)
    if digest != cfg["manifest_sha256"]:
        raise ValueError("Eğitim manifesti B'den sonra değişmiş.")
    out.mkdir(parents=True, exist_ok=True)
    config = {k:v for k,v in vars(a).items() if k not in ("root","cache_dir","resume")}
    config.update(version=VERSION, b_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                  b_config=cfg, manifest_sha256=digest, torch=str(torch.__version__),
                  transformers=transformers.__version__, peft=peft.__version__, lora_rank=16, lora_alpha=32,
                  lora_targets=["q_proj","v_proj"], reward="exact_match_plus_piecewise_wer", word_score_floor=0.0)
    state = torch.load(last, map_location="cpu", weights_only=True) if a.resume else None
    # Eski checkpoint'ler harf duyarlıydı; yeni ölçütle aynı koşuya devam edilmez.
    if state and "em_ignore_case" not in state["config"]:
        state["config"]["em_ignore_case"] = False
        if state["config"]["version"] == "1.0.0":
            state["config"]["version"] = VERSION
    if state and state["config"] != config:
        raise ValueError("Devam ayarları veya B checkpoint'i farklı.")
    (out/"config.json").write_text(json.dumps(config, indent=2))
    dtype = torch.bfloat16
    torch.manual_seed(a.seed)
    feature = WhisperFeatureExtractor.from_pretrained(cfg["whisper"],revision=cfg["whisper_revision"])
    w = WhisperModel.from_pretrained(cfg["whisper"],revision=cfg["whisper_revision"])
    width = w.config.d_model
    encoder = w.encoder.to("cuda",dtype).eval().requires_grad_(False)
    del w
    tok = AutoTokenizer.from_pretrained(cfg["qwen"],revision=cfg["qwen_revision"])
    lm, info = Qwen3_5ForCausalLM.from_pretrained(cfg["qwen"],revision=cfg["qwen_revision"],
                                                dtype=dtype,output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError("Dil modeli ağırlıkları eksik.")
    lm = lm.cuda().eval().requires_grad_(False)
    embed = lm.get_input_embeddings()
    proj = nn.Sequential(nn.Linear(width*5,2048),nn.ReLU(),nn.Linear(2048,embed.embedding_dim)).cuda()
    proj.load_state_dict(torch.load(src,map_location="cpu",weights_only=True))
    # Sesi metin uzayına taşıyan B projector'ı artık sabittir. Qwen'in tam
    # dikkat bloklarındaki Q ve V dönüşümlerine düşük-rank ek matrisler öğrenilir.
    # Aynı LM üzerinde adaptörü geçici kapatmak değişmeyen B politikasını verir.
    proj.eval().requires_grad_(False)
    adapter_config = LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
                                target_modules=["q_proj", "v_proj"], lora_dropout=0., bias="none")
    lm = get_peft_model(lm, adapter_config)
    lm.eval()  # Gradyanı kapatmaz; örnekleme ve puanlamada dropout kapalıdır.
    params = [p for p in lm.parameters() if p.requires_grad]
    if not params or any("lora_" not in n for n,p in lm.named_parameters() if p.requires_grad):
        raise RuntimeError("Eğitilebilir parametreler LoRA ile sınırlı olmalı.")
    print("Eğitilebilir Qwen-LoRA parametresi:", sum(p.numel() for p in params), flush=True)
    adapter_config.save_pretrained(out)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.)
    step, best = 0, float("inf")
    if state:
        set_peft_model_state_dict(lm, state["adapter"])
        opt.load_state_dict(state["optimizer"])
        step, best = state["step"],state["best"]
        del state

    def save(name,obj):
        target=out/name
        tmp=target.with_suffix(".tmp")
        torch.save(obj,tmp)
        tmp.replace(target)

    def checkpoint():
        save("last.pt",dict(config=config,step=step,best=best,adapter=get_peft_model_state_dict(lm),
                            optimizer=opt.state_dict()))

    def log(**data):
        print(data,flush=True)
        with (out/"metrics.jsonl").open("a") as f:
            f.write(json.dumps(data,ensure_ascii=False)+"\n")

    def emb(ids):
        return embed(torch.tensor(ids,device="cuda",dtype=torch.long))

    bos=[] if tok.bos_token_id is None else [tok.bos_token_id]
    prefix=emb(bos+tok.encode("USER: ",add_special_tokens=False))
    prompt=emb(tok.encode(" Transcribe speech to text. ASSISTANT:",add_special_tokens=False))
    eos=tok.eos_token_id

    @torch.no_grad()
    def frames(wave):
        x=feature([wave],sampling_rate=16000,return_tensors="pt").input_features
        h=encoder(x.to("cuda",dtype)).last_hidden_state[0]
        return h[:len(h)//5*5].reshape(-1,5*width).float()

    def context(h, module):
        return torch.cat((prefix,module(h).to(dtype),prompt))

    def generate(ctx, count, sampling, limit):
        # Sıcaklık 1, kesmesiz kategorik örnekleme: daha sonra hesaplanan log p
        # ile adayları üreten dağılım eşleşir. Model deposunun varsayılan top-k
        # veya tekrar cezasını yanlışlıkla devralmamak için açık config kullanılır.
        gen=GenerationConfig(do_sample=sampling, num_beams=1, num_return_sequences=count,
                             max_new_tokens=limit, eos_token_id=eos, pad_token_id=eos,
                             bos_token_id=tok.bos_token_id, top_k=0 if sampling else 50)
        x=ctx.unsqueeze(0)
        mask=torch.ones(x.shape[:2],device=x.device,dtype=torch.long)
        seqs=lm.generate(inputs_embeds=x,attention_mask=mask,generation_config=gen)
        result=[]
        for seq in seqs.tolist():
            # İlk EOS hedefe dahildir. Batch'teki daha uzun adaylar nedeniyle
            # sona gelen dolgu EOS'ları ödüle ve politika kaybına katılmaz.
            result.append(seq[:seq.index(eos)+1] if eos in seq else seq)
        return result

    def log_probs(ctx, seq):
        # İlk üretilen token, bağlamın SON konumundaki logit ile puanlanır.
        # Sadece üretilmiş token'lar kullanılır. Referans metni girdiye girmez.
        ids=torch.tensor(seq,device="cuda")
        x=torch.cat((ctx,embed(ids))).unsqueeze(0)
        logits=lm(inputs_embeds=x,use_cache=False).logits[0,len(ctx)-1:-1].float()
        return logits.log_softmax(-1).gather(1,ids[:,None]).squeeze(1)

    def group(path, text):
        wave,rate=sf.read(path,dtype="float32")
        if rate!=16000 or wave.ndim!=1 or not 0<len(wave)<=480000:
            raise ValueError(f"06 manifestiyle uyuşmayan ses: {path}")
        h=frames(wave)
        with torch.no_grad():
            old=context(h,proj)
            candidates=generate(old,a.group_size,True,a.max_new_tokens)
            hyps=tok.batch_decode(candidates,skip_special_tokens=True)
            values = [reward(text, hyp, dev.errors, a) for hyp in hyps]
            # ED tam sayı döndürür. Ortalama ve standart sapma için tüm ödülleri float32 yap.
            rewards=torch.tensor(values,device="cuda",dtype=torch.float32)
            sd=rewards.std(unbiased=False)
            advantage=(rewards-rewards.mean())/(sd+1e-6)
        for i,seq in enumerate(candidates):
            # Tek rollout için bir optimizer güncellemesi yapıyoruz. Eski politika
            # log olasılığı sabitlenir; yeni politika aynı noktadan türev alır.
            with torch.no_grad():
                prev=log_probs(old,seq)
                with lm.disable_adapter():
                    base=log_probs(old,seq)
            now=log_probs(old,seq)
            # Arada optimizer adımı yoktur: oran 1 civarında başlar. Bu tek geçişte
            # kırpma etkin bir sınır oluşturmaz, ancak now üzerinden gradyan akar.
            ratio=(now-prev).exp()
            surrogate=torch.minimum(ratio*advantage[i],ratio.clamp(1-a.clip,1+a.clip)*advantage[i])
            d=base-now
            kl=torch.expm1(d)-d
            # Önce adayın token ortalaması, ardından aday ve ses ortalaması.
            loss=(-surrogate+a.beta*kl).mean()/(a.group_size*a.sounds_per_step)
            if not torch.isfinite(loss):
                raise FloatingPointError("Sonlu olmayan GRPO kaybı.")
            loss.backward()
        return values, float(sd), hyps

    @torch.no_grad()
    def evaluate(sample, work):
        counts=dict(N=0,S=0,D=0,I=0)
        for row in sample:
            w,sr=sf.read(work/(row["id"]+".flac"),dtype="float32")
            parts=[]
            for chunk in dev.audio_chunks(w,sr,row["id"]):
                seq=generate(context(frames(chunk),proj),1,False,cfg["max_new_tokens"])[0]
                parts.append(tok.decode(seq,skip_special_tokens=True))
            c=dev.errors(row["text"]," ".join(parts))
            for k in counts:
                counts[k]+=c[k]
        return 100*(counts["S"]+counts["D"]+counts["I"])/counts["N"],counts

    if cfg["num_beams"]!=1:
        raise ValueError("Bu D sürümü B'nin greedy geliştirme protokolünü bekliyor.")
    with tempfile.TemporaryDirectory(dir=a.cache_dir) as tmp:
        work=Path(tmp)
        sample=dev.prepare(a.root/"data/archives/dev-other.tar.gz",cfg["dev_samples"],cfg["seed"],work,out)
        if not a.resume:
            best,counts=evaluate(sample,work)
            log(step=0,dev_wer_percent=best,**counts)
            save("best_adapter.pt",get_peft_model_state_dict(lm))
            checkpoint()
        epoch=-1
        while step<a.steps:
            start=time.monotonic()
            # Her adımın örnekleme tohumu ve veri sırası yeniden kurulabilir.
            # Kesinti sonrası birikmemiş gradyanlar son checkpoint'ten tekrar hesaplanır.
            torch.manual_seed(a.seed+step)
            opt.zero_grad(set_to_none=True)
            reward_sum=0.
            zero_groups=0
            em_count=0
            candidate_count=0
            varying=0
            for j in range(a.sounds_per_step):
                e,pos=divmod(step*a.sounds_per_step+j,len(rows))
                if e!=epoch:
                    order=list(range(len(rows)))
                    random.Random(a.seed+e).shuffle(order)
                    epoch=e
                path,text=rows[order[pos]]
                values,sd,hyps=group(path,text)
                zero_groups += int(all(value == 0 for value in values))
                # Aynı adayın tekrar örneklenmesi de üretim sayısına dahildir.
                em_count += sum(exact_match(text, hyp, a.em_ignore_case) for hyp in hyps)
                candidate_count += len(hyps)
                varying+=int(sd>0)
                reward_sum+=sum(values)/len(values)
                if step==0:
                    log(example=path.stem,reference=text,candidates=hyps,rewards=values)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.step()
            step+=1
            log(step=step,em_count=em_count,candidate_count=candidate_count,em_rate=em_count/candidate_count,
                mean_reward=reward_sum/a.sounds_per_step,varying_groups=varying,
                zero_group_fraction=zero_groups/a.sounds_per_step,
                varying_group_fraction=varying/a.sounds_per_step,
                grad_norm=float(norm),train_seconds=time.monotonic()-start)
            measured=step%a.eval_every==0 or step==a.steps
            if measured:
                wer,counts=evaluate(sample,work)
                log(step=step,dev_wer_percent=wer,**counts)
                if wer<best:
                    best=wer
                    save("best_adapter.pt",get_peft_model_state_dict(lm))
            if measured or step%a.save_every==0:
                checkpoint()
        save("adapter.pt",get_peft_model_state_dict(lm))
        print(f"Bitti: {out}, en iyi WER %{best:.2f}",flush=True)


if __name__=="__main__":
    main()
