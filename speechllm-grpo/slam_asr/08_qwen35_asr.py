"""YZM5110 • Deney B/E • Sürüm 1.1.0
LibriSpeech-960 ile Whisper-small → projector → Qwen3.5-Base (0.8B veya 9B) denetimli eğitimi.
06'nın train-960.jsonl çıktısını okur. Geliştirme WER'i için yanındaki
05_compare_dev_other.py dosyasının örnek seçimi, bölümleme ve WER kodunu kullanır.
--help yalnızca Python standart kütüphanesini gerektirir.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import shutil
import tarfile
import tempfile
import time

VERSION = "1.1.0"
WHISPER = "openai/whisper-small"
QWEN = "Qwen/Qwen3.5-0.8B-Base"
SPLITS = {"train-clean-100", "train-clean-360", "train-other-500"}


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # QWEN sabiti, 0.8B kullanan GRPO betikleri için de varsayılan modeldir.
    p.add_argument("--qwen", choices=[QWEN, "Qwen/Qwen3.5-9B-Base"], default=QWEN,
                   help="Donuk dil modeli. Projector çıkışı embedding genişliğinden belirlenir.")
    p.add_argument("--version", action="version", version=VERSION)
    p.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/slam_asr"), help="Kalıcı proje dizini")
    p.add_argument("--cache-dir", type=Path, default=Path("/content/slam_asr_audio"), help="Açılmış FLAC dosyalarının yerel dizini")
    p.add_argument("--prepare-only", action="store_true", help="Yalnızca eğitim seslerini hazırla")
    p.add_argument("--mode", choices=["pilot", "train"], required=True, help="İki sesle bağlantı kontrolü veya tam SFT")
    p.add_argument("--run-name", default=None, help="runs içinde tek koşu için klasör adı")
    p.add_argument("--resume", action="store_true", help="Aynı koşunun last.pt dosyasından sürdür")
    for name, default, desc in [
        ("batch-size", 4, "Tek ileri geçişteki ses sayısı"),
        ("gradient-accumulation-steps", 1, "Bir optimizer güncellemesindeki mikro-batch sayısı"),
        ("warmup-steps", 1000, "LR'nin doğrusal arttığı güncelleme sayısı"),
        ("max-steps", 100000, "Toplam optimizer güncellemesi üst sınırı"),
        ("save-every", 500, "Devam checkpoint'i yazma aralığı"),
        ("log-every", 25, "Eğitim kaybı yazma aralığı"),
        ("eval-every", 2000, "Geliştirme WER ölçümleri arası güncelleme sayısı"),
        ("dev-samples", 64, "Sabit dev-other altkümesinin büyüklüğü"),
        ("patience", 5, "İyileşmesiz ölçüm sınırı; 0 erken durdurmayı kapatır"),
        ("num-beams", 1, "Geliştirme ölçümünde arama genişliği"),
        ("max-new-tokens", 160, "Her ses parçasında üretim sınırı"),
        ("seed", 42, "Başlatma, örnek seçimi ve karıştırma tohumu"),
    ]:
        p.add_argument("--" + name, type=int, default=default, help=desc)
    p.add_argument("--lr", "--learning-rate", type=float, default=1e-4, help="Isınma sonrasındaki sabit öğrenme oranı")
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW ağırlık azaltma katsayısı")
    a = p.parse_args()
    if a.run_name is None:
        a.run_name = ("B_qwen08_" if a.qwen == QWEN else "E_qwen9b_") + a.mode
    if a.mode == "pilot" and a.resume:
        p.error("Pilot tek güncellemeliktir; tekrar için yeni koşu adı seçin.")
    for key, value in vars(a).items():
        if type(value) is int and value < (0 if key in {"warmup_steps", "seed", "patience"} else 1):
            p.error(f"Geçersiz değer: {key}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", a.run_name):
        p.error("Koşu adı harf, rakam, alt çizgi veya tire içermeli.")
    if not (math.isfinite(a.lr) and a.lr > 0 and math.isfinite(a.weight_decay) and a.weight_decay >= 0):
        p.error("lr pozitif, weight-decay sıfır veya pozitif ve her ikisi de sonlu olmalı.")
    return a


def train_data(a):
    manifest = a.root / "manifests/train-960.jsonl"
    digest = hashlib.sha256()
    rows, seen = [], set()
    groups = defaultdict(dict)
    with manifest.open("rb") as src:
        for line in src:
            digest.update(line)
            r = json.loads(line)
            uid, split = r["id"], r["split"]
            if not re.fullmatch(r"\d+-\d+-\d+", uid) or uid in seen or split not in SPLITS:
                raise ValueError(f"Geçersiz split veya kayıt kimliği: {uid}")
            speaker, chapter, _ = uid.split("-")
            member = f"LibriSpeech/{split}/{speaker}/{chapter}/{uid}.flac"
            if (member != r["member"] or r["archive"] != f"data/archives/{split}.tar.gz"
                    or not 0 < r["duration_seconds"] <= 30 or not r["text"].strip()):
                raise ValueError(f"06 manifestindeki yol, süre veya metni kontrol edin: {uid}")
            seen.add(uid)
            path = a.cache_dir / member
            groups[split][member] = path
            rows.append((path, r["text"]))
            # Pilot için yalnızca iki manifest satırını ve ilgili iki FLAC'ı oku.
            if a.mode == "pilot" and len(rows) == 2:
                break
    if a.mode == "pilot" and len(rows) != 2:
        raise ValueError("Pilot için iki kayıt gerekli.")
    if a.mode == "train" and (set(groups) != SPLITS or len(rows) != 281241):
        raise ValueError(f"Tam eğitim için üç bölüm ve 281241 kayıt bekleniyor: {len(rows)}")
    # Sıkıştırılmış tar üzerinde rastgele kayıt aramak pahalıdır. Arşivi sırayla
    # bir kez dolaşırız; sonraki epoch'larda karıştırılmış FLAC yollarını okuruz.
    for split, paths in groups.items():
        stage(a.root / f"data/archives/{split}.tar.gz", paths)
    return rows, digest.hexdigest()


def stage(archive, paths):
    missing = {m: p for m, p in paths.items() if not p.is_file()}
    if not missing:
        return
    print(f"{archive.name}: {len(missing)} ses yerel diske hazırlanıyor.", flush=True)
    with tarfile.open(archive, "r|gz") as tar:
        for item in tar:
            if item.name not in missing or not item.isfile():
                continue
            dest = missing.pop(item.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Disk dolarsa yarım FLAC bir sonraki çalışmada tamamlanmış sayılmaz.
            tmp = dest.with_suffix(".partial")
            with tar.extractfile(item) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(dest)
            if not missing:
                break
    if missing:
        raise FileNotFoundError(f"{archive}: {next(iter(missing))} bulunamadı.")


def main():
    a = arguments()
    rows, digest = train_data(a)
    if a.prepare_only:
        print(f"Hazır: {len(rows)} eğitim kaydı.")
        return

    import soundfile as sf
    import torch
    from torch import nn
    from huggingface_hub import hf_hub_download
    import transformers
    from transformers import Qwen3_5ForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Bu eğitim yapılandırması BF16 destekli CUDA GPU gerektirir.")
    # 05 ile aynı normalizasyon ve sözcük hata sayımı kullanılır.
    spec = importlib.util.spec_from_file_location("dev_eval", Path(__file__).with_name("05_compare_dev_other.py"))
    dev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dev)
    out = a.root / "runs" / a.run_name
    last = out / "last.pt"
    if a.resume and not last.is_file():
        raise FileNotFoundError(last)
    if not a.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Dolu koşu klasörü: {out}. Checkpoint varsa --resume kullanın veya yeni ad seçin.")
    out.mkdir(parents=True, exist_ok=True)
    print("Koşu:", out, flush=True)
    state = torch.load(last, map_location="cpu", weights_only=True) if a.resume else None
    old = state["config"] if state else {}
    if state and old["qwen"] != a.qwen:
        raise ValueError("Checkpoint ve --qwen farklı modellere ait.")
    torch.manual_seed(a.seed)
    dtype = torch.bfloat16
    # Hub config dosyasını snapshots/<commit>/ altında önbelleğe alır.
    # resolve() dosyanın blobs/ hedefini izlerdi; burada snapshot yolunu koruyoruz.
    def revision(repo, saved):
        if saved:
            return saved
        config_path = Path(hf_hub_download(repo, "config.json"))
        commit = config_path.parent.name
        if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            raise RuntimeError(f"{repo}: snapshot sürümü okunamadı: {config_path}")
        return commit

    wr = revision(WHISPER, old.get("whisper_revision"))
    vr = revision(a.qwen, old.get("qwen_revision"))
    feature = WhisperFeatureExtractor.from_pretrained(WHISPER, revision=wr)
    whisper = WhisperModel.from_pretrained(WHISPER, revision=wr)
    enc = whisper.encoder.to("cuda", dtype).eval().requires_grad_(False)
    whisper_width = whisper.config.d_model
    del whisper
    tok = AutoTokenizer.from_pretrained(a.qwen, revision=vr)
    # Dil omurgası ve sözcük çıkış katmanı yüklenir. Görsel encoder kullanılmaz.
    # Eksik dil ağırlıkları rastgele başlatılırsa deney geçersiz olur.
    llm, info = Qwen3_5ForCausalLM.from_pretrained(
        a.qwen, revision=vr, dtype=dtype, output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise RuntimeError(f"Qwen dil ağırlıkları tam yüklenemedi: {info}")
    if tok.eos_token_id is None:
        raise ValueError("Bitiş token kimliği bulunamadı.")
    llm = llm.cuda().eval().requires_grad_(False)
    emb = llm.get_input_embeddings()
    # Projector FP32 kalır. Başlangıç ağırlıkları yeni üretilir; pilot sonuçları
    # hiperparametre seçimini bilgilendirir, tam eğitimin başlangıcına taşınmaz.
    proj = nn.Sequential(nn.Linear(whisper_width * 5, 2048), nn.ReLU(),
                         nn.Linear(2048, emb.embedding_dim)).cuda()
    opt = torch.optim.AdamW(proj.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    # LambdaLR ilk optimizer adımından önce katsayıyı uygular. Böylece 1. adım
    # lr/1000, 1000. adım lr kullanır. Sonra katsayı 1 olarak kalır.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1., (k + 1) / max(1, a.warmup_steps)))
    cfg = {k: v for k, v in vars(a).items() if k not in {"root","cache_dir","resume","prepare_only"}}
    cfg.update(version=VERSION, manifest_sha256=digest, whisper=WHISPER, qwen=a.qwen,
               whisper_revision=wr, qwen_revision=vr, dtype=str(dtype),
               torch=str(torch.__version__), transformers=transformers.__version__)
    step = epoch = offset = stale = 0
    best = float("inf")
    if state:
        changed = [k for k in cfg if old.get(k) != cfg[k] and k not in {"save_every","log_every","max_steps","version"}]
        if changed:
            raise ValueError(f"Devam ayarları checkpoint ile uyuşmuyor: {changed}")
        proj.load_state_dict(state["projector"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step, epoch, offset, best, stale = (state[k] for k in ("step","epoch","offset","best","stale"))
        del state
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def embed(ids):
        return emb(torch.tensor(ids, device="cuda"))

    def ids(s):
        return tok.encode(s, add_special_tokens=False)

    # Base modelde sohbet şablonu kullanmıyoruz. BOS tanımsızsa yapay bir
    # özel token eklemeyiz. Ses bağlamını her iki modda aynı komut izler.
    prefix = embed(([] if tok.bos_token_id is None else [tok.bos_token_id]) + ids("USER: "))
    prompt = embed(ids(" Transcribe speech to text. ASSISTANT:"))

    def contexts(waves):
        # Whisper sabit 30 saniyelik pencerede 1500 × 768 temsil üretir.
        # Pilotla aynı şekilde padding dahil beş komşu çerçeveyi birleştiririz:
        # [B,1500,768] → [B,300,3840]. Bu işlem ortalama alma değildir.
        with torch.no_grad():
            x = feature(waves, sampling_rate=16000, return_tensors="pt").input_features
            frames = enc(x.to("cuda", dtype)).last_hidden_state
            b, t, h = frames.shape
            grouped = frames[:, :t//5*5].reshape(b, t//5, 5*h)
        # no_grad burada biter: gradyan projector'a kadar geri akmalı.
        sounds = proj(grouped.float()).to(dtype)
        return [torch.cat((prefix, s, prompt)) for s in sounds]

    def micro_loss(indices):
        waves, targets = [], []
        for i in indices:
            path, text = rows[i]
            w, rate = sf.read(path, dtype="float32")
            if rate != 16000 or w.ndim != 1 or not 0 < len(w) <= 480000:
                raise ValueError(f"16 kHz mono ve en fazla 30 saniye bekleniyor: {path}")
            waves.append(w)
            targets.append(ids(" " + text) + [tok.eos_token_id])
        inputs, labels = [], []
        for ctx, target in zip(contexts(waves), targets):
            inputs.append(torch.cat((ctx, embed(target))))
            # Öğretmen zorlaması: model önceki doğru sözcük parçalarını görür.
            # -100 bağlamı ve dolguyu kayıptan çıkarır. CausalLM kaydırmayı
            # içeride yapar; ilk hedef token son bağlam konumundan tahmin edilir.
            labels.append(torch.tensor([-100]*len(ctx) + target, device="cuda"))
        pad = nn.utils.rnn.pad_sequence
        batch = pad(inputs, batch_first=True)
        lengths = torch.tensor([len(s) for s in inputs], device="cuda")
        mask = torch.arange(batch.size(1), device="cuda")[None, :] < lengths[:, None]
        loss = llm(inputs_embeds=batch, attention_mask=mask,
                   labels=pad(labels, batch_first=True, padding_value=-100), use_cache=False).loss
        return loss, sum(map(len, targets))

    def save(obj, name):
        path = out / name
        tmp = path.with_suffix(".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)

    def checkpoint():
        # offset, sıradaki örneğin epoch içindeki konumudur. Yalnızca tamamlanmış
        # optimizer adımlarını kaydederiz; yarım birikmiş gradyan taşınmaz.
        save(dict(config=cfg, projector=proj.state_dict(), optimizer=opt.state_dict(),
                  scheduler=sched.state_dict(), step=step, epoch=epoch, offset=offset,
                  best=best, stale=stale), "last.pt")

    def log(record):
        print(record, flush=True)
        with (out / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    @torch.no_grad()
    def decode(wave):
        # Bu fonksiyon referans metni almaz. Eğitimdeki doğru önceki token'ların
        # yerini üretim sırasında modelin kendi tahminleri alır.
        ctx = contexts([wave])[0].unsqueeze(0)
        mask = torch.ones(ctx.shape[:2], device=ctx.device, dtype=torch.long)
        result = llm.generate(inputs_embeds=ctx, attention_mask=mask,
                              do_sample=False, num_beams=a.num_beams,
                              max_new_tokens=a.max_new_tokens, pad_token_id=tok.eos_token_id)
        return result[0].tolist()

    if a.mode == "pilot":
        # Pilotun tek güncellemesi komuttaki lr ile yapılır. Tam eğitimin
        # warmup planı ayrı koşuda, yeni bir projector ile başlar.
        opt.param_groups[0]["lr"] = a.lr
        initial = proj[0].weight.detach().clone()
        loss, tokens = micro_loss([0, 1])
        if not torch.isfinite(loss):
            raise FloatingPointError("Pilot kaybı sonlu değil.")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(proj.parameters(), float("inf"), error_if_nonfinite=True)
        if norm.item() == 0 or any(p.grad is None for p in proj.parameters()):
            raise RuntimeError("Projector'a beklenen gradyan ulaşmadı.")
        opt.step()
        delta = (proj[0].weight.detach() - initial).abs().max().item()
        if not delta > 0:
            raise RuntimeError("İlk katmanın ağırlıkları değişmedi.")
        waves = [sf.read(path, dtype="float32")[0] for path, _ in rows]
        predicted = [decode(w) for w in waves]
        save(proj.state_dict(), "projector.pt")
        # Yeni modüle diskten yükleyerek aynı iki ses için token dizisini sınarız.
        import copy
        proj = copy.deepcopy(proj)
        proj.load_state_dict(torch.load(out/"projector.pt", weights_only=True, map_location="cpu"))
        if predicted != [decode(w) for w in waves]:
            raise RuntimeError("Yeniden yüklemeden sonra token dizisi değişti.")
        for (path, ref), seq in zip(rows, predicted):
            log(dict(id=path.stem, reference=ref, hypothesis=tok.decode(seq, skip_special_tokens=True)))
        log(dict(status="PASS", batch_loss=loss.item(), supervised_tokens=tokens,
                 grad_norm=norm.item(), max_delta=delta))
        return

    @torch.no_grad()
    def evaluate(sample, work):
        counts = dict(N=0, S=0, D=0, I=0)
        for r in sample:
            w, rate = sf.read(work / (r["id"]+".flac"), dtype="float32")
            parts = []
            for chunk in dev.audio_chunks(w, rate, r["id"]):
                parts.append(tok.decode(decode(chunk), skip_special_tokens=True))
            c = dev.errors(r["text"], " ".join(parts))
            for k in counts:
                counts[k] += c[k]
        # Kayıt bazındaki yüzdelerin ortalaması değil, tüm referans kelimeleri
        # üzerindeki ekleme + silme + değiştirme sayısı kullanılır.
        return 100 * (counts["I"] + counts["D"] + counts["S"]) / counts["N"], counts

    with tempfile.TemporaryDirectory(dir=a.cache_dir) as tmp:
        work = Path(tmp)
        sample = dev.prepare(a.root/"data/archives/dev-other.tar.gz", a.dev_samples, a.seed, work, out)
        # Karıştırma her epoch için bağımsız bir tohumdan türetilir. Devam ederken
        # aynı sıra yeniden kurulur ve kaydedilen offset'e doğrudan geçilir.
        order = None
        if not a.resume:
            checkpoint()
        while step < a.max_steps and (not a.patience or stale < a.patience):
            if order is None:
                order = list(range(len(rows)))
                random.Random(a.seed + epoch).shuffle(order)
            end = min(len(rows), offset + a.batch_size * a.gradient_accumulation_steps)
            tokens = 0
            summed = 0.
            opt.zero_grad(set_to_none=True)
            for start in range(offset, end, a.batch_size):
                loss, n = micro_loss(order[start:min(start + a.batch_size, end)])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Sonlu olmayan kayıp, güncelleme {step+1}")
                # Farklı uzunluktaki mikro-batch'leri hedef token sayısıyla tart.
                (loss * n).backward()
                summed += loss.item() * n
                tokens += n
            for p in proj.parameters():
                p.grad.div_(tokens)
            nn.utils.clip_grad_norm_(proj.parameters(), float("inf"), error_if_nonfinite=True)
            lr_used = opt.param_groups[0]["lr"]
            opt.step()
            sched.step()
            step += 1
            offset = end
            if offset == len(rows):
                epoch += 1
                offset = 0
                order = None
            if step == 1 or step % a.log_every == 0:
                log(dict(step=step, completed_epochs=epoch, next_offset=offset, batch_loss=summed/tokens,
                         lr=lr_used, tokens=tokens, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                         unix_time=time.time()))
            measured = step % a.eval_every == 0 or step == a.max_steps
            if measured:
                wer, counts = evaluate(sample, work)
                log(dict(step=step, dev_wer_percent=wer, **counts))
                if wer < best:
                    best, stale = wer, 0
                    save(proj.state_dict(), "best_projector.pt")
                else:
                    stale += 1
            if measured or step % a.save_every == 0:
                checkpoint()
        save(proj.state_dict(), "projector.pt")
        print(f"Bitti. Adım {step}, en iyi geliştirme WER %{best:.2f}. Çıktılar: {out}")


if __name__ == "__main__":
    main()
