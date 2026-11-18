"""YZM5110 | 64 kayıtlık Vicuna pilotu | Sürüm 1.1.0
Bu sürüm: Eğitim ve üretim ayarları komut satırından alınır.
Pilot: sabit örneklerde serbest üretim, projector eğitimi ve yeniden yükleme.
Girdi: manifests/pilot_64_seed42.jsonl ve FLAC dosyaları.
Çıktı: runs altında her çağrı için ayrı dizin, epoch günlükleri ve checkpoint.
"""
import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import math

WHISPER = "openai/whisper-small"
VICUNA = "lmsys/vicuna-7b-v1.5"


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/slam_asr"), help="Drive proje dizini")
    parser.add_argument("--epochs", type=int, default=5, help="64 kayıt üzerinde tam geçiş sayısı")
    parser.add_argument("--batch-size", type=int, default=4, help="Bir optimizer adımındaki kayıt sayısı")
    parser.add_argument("--lr", "--learning-rate", type=float, default=1e-4, help="AdamW öğrenme oranı")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW ağırlık azaltma katsayısı")
    parser.add_argument("--seed", type=int, default=42, help="Ağırlık başlatma ve kayıt sıralama tohumu")
    parser.add_argument("--eval-examples", type=int, default=4, help="Önce/sonra karşılaştırılacak sabit eğitim kaydı sayısı")
    parser.add_argument("--num-beams", type=int, default=1, help="Üretimde tutulan aday sayısı; 1 açgözlü çözümleme")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Üretilen transkriptin token sınırı")
    args = parser.parse_args()
    if min(args.epochs, args.num_beams, args.max_new_tokens) < 1:
        parser.error("Epoch, beam ve token sınırı pozitif olmalı.")
    if not (1 <= args.batch_size <= 64 and 1 <= args.eval_examples <= 64):
        parser.error("Batch boyutu ve örnek sayısı 1–64 aralığında olmalı.")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Öğrenme oranı pozitif, weight decay sıfır veya pozitif ve ikisi de sonlu olmalı.")
    if not 0 <= args.seed < 2**63:
        parser.error("Seed 0 ile 2^63-1 arasında olmalı.")
    return args


def main():
    args = parse_args()
    # Yardım ve argüman kontrolleri model kütüphanelerini yüklemeden çalışır.
    import soundfile as sf
    import torch
    import transformers
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

    if not torch.cuda.is_available():
        raise RuntimeError("GPU etkin bir Colab oturumu gerekiyor.")
    torch.manual_seed(args.seed)
    order_rng = random.Random(args.seed)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # Manifest sabittir: --seed mevcut 64 kaydı değiştirmez.
    manifest = args.root / "manifests/pilot_64_seed42.jsonl"
    raw = manifest.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    if len(rows) != 64 or len({r["id"] for r in rows}) != 64:
        raise ValueError("Manifest 64 benzersiz kayıt içermeli.")
    out = args.root / "runs" / datetime.now(timezone.utc).strftime("A_pilot_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    print("Çıktı dizini:", out, flush=True)

    # Dondurulmuş encoder her epoch aynı temsili üretir. 64 kaydı bir kez kodla.
    processor = WhisperFeatureExtractor.from_pretrained(WHISPER)
    whisper = WhisperModel.from_pretrained(WHISPER)
    whisper_revision = whisper.config._commit_hash
    encoder = whisper.encoder.to("cuda", dtype).eval().requires_grad_(False)
    del whisper
    cached = []
    with torch.no_grad():
        for row in rows:
            wave, rate = sf.read(args.root / row["audio_path"], dtype="float32")
            if rate != 16000 or wave.ndim != 1 or not 0 < len(wave) <= 480000:
                raise ValueError(f"Beklenmeyen ses biçimi veya süre: {row['id']}")
            features = processor(wave, sampling_rate=rate, return_tensors="pt").input_features
            frames = encoder(features.to("cuda", dtype)).last_hidden_state[0]
            steps, width = frames.shape
            # Beş zaman çerçevesini tek uzun vektöre çevirip CPU belleğinde tut.
            cached.append(frames[:steps // 5 * 5].reshape(-1, width * 5).cpu())
    del encoder
    tok = AutoTokenizer.from_pretrained(VICUNA)
    llm = AutoModelForCausalLM.from_pretrained(VICUNA, torch_dtype=dtype).cuda().eval()
    # LLM ağırlıkları donuk; eğitimde gradyan yine LLM girdisinden projector’a akar.
    llm.requires_grad_(False)
    embed = llm.get_input_embeddings()

    def make_projector():
        return nn.Sequential(nn.Linear(width * 5, 2048), nn.ReLU(),
                             nn.Linear(2048, embed.embedding_dim)).cuda()

    def token_ids(text):
        return tok.encode(text, add_special_tokens=False)

    def embeddings(ids):
        return embed(torch.tensor(ids, device="cuda"))

    prefix = embeddings([tok.bos_token_id] + token_ids("USER: "))
    prompt = embeddings(token_ids(" Transcribe speech to text. ASSISTANT:"))
    projector = make_projector()
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    config = dict(version="1.1.0", seed=args.seed, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr,
                  weight_decay=args.weight_decay, eval_examples=args.eval_examples,
                  num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                  whisper=WHISPER, vicuna=VICUNA, whisper_revision=whisper_revision,
                  vicuna_revision=llm.config._commit_hash, torch=str(torch.__version__),
                  transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
                  dtype=str(dtype), manifest_sha256=hashlib.sha256(raw).hexdigest())
    save_json(out / "config.json", config)

    def contexts(indices, module):
        sounds = module(torch.stack([cached[i] for i in indices]).cuda().float()).to(dtype)
        return [torch.cat([prefix, sound, prompt]) for sound in sounds]

    @torch.no_grad()
    def transcribe(module):
        # Manifestin ilk eval_examples kaydı her iki ölçümde de sabit. Üretim bağlamına referans metni eklenmez.
        predictions = []
        for i, context in enumerate(contexts(range(args.eval_examples), module)):
            result = llm.generate(inputs_embeds=context.unsqueeze(0),
                                  attention_mask=torch.ones((1, len(context)), device="cuda", dtype=torch.long),
                                  do_sample=False, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                                  pad_token_id=tok.eos_token_id)
            predictions.append(dict(id=rows[i]["id"], reference=rows[i]["text"],
                                    hypothesis=tok.decode(result[0], skip_special_tokens=True)))
        return predictions

    before = transcribe(projector)
    save_json(out / "before.json", before)
    for epoch in range(1, args.epochs + 1):
        indices = list(range(len(rows)))
        order_rng.shuffle(indices)
        total_loss = token_count = 0
        start = time.monotonic()
        for offset in range(0, len(indices), args.batch_size):
            batch = indices[offset:offset + args.batch_size]
            sequences, labels = [], []
            for i, context in zip(batch, contexts(batch, projector)):
                ids = token_ids(" " + rows[i]["text"]) + [tok.eos_token_id]
                sequences.append(torch.cat([context, embeddings(ids)]))
                # Sadece transkript ve bitiş tokenı kayba katılır.
                labels.append(torch.tensor([-100] * len(context) + ids, device="cuda"))
            pad = nn.utils.rnn.pad_sequence
            inputs = pad(sequences, batch_first=True)
            targets = pad(labels, batch_first=True, padding_value=-100)
            lengths = torch.tensor([len(s) for s in sequences], device="cuda")
            # Sağdaki padding konumları gerçek girdi olmadığı için attention maskesinde kapatılır.
            mask = torch.arange(inputs.size(1), device="cuda")[None, :] < lengths[:, None]
            optimizer.zero_grad(set_to_none=True)
            loss = llm(inputs_embeds=inputs, labels=targets, attention_mask=mask, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Sonlu olmayan kayıp")
            loss.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in projector.parameters()):
                raise ValueError("Projector gradyanı geçersiz")
            optimizer.step()
            n = (targets != -100).sum().item()
            # Epoch kaybını hedef token sayısıyla ağırlıklandır; son kısa batch doğru katkı verir.
            total_loss += loss.item() * n
            token_count += n
        metrics = dict(epoch=epoch, loss=total_loss/token_count, seconds=time.monotonic()-start,
                       peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
        print(metrics, flush=True)
        with (out / "epochs.jsonl").open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        # Tam yazılan son checkpoint’i tek işlemde hedef ada taşı.
        temp = out / "projector.tmp"
        torch.save(projector.state_dict(), temp)
        temp.replace(out / "projector.pt")
    after = transcribe(projector)
    save_json(out / "after.json", after)
    restored = make_projector()
    restored.load_state_dict(torch.load(out / "projector.pt", weights_only=True))
    if transcribe(restored) != after:
        raise RuntimeError("Checkpoint yeniden yüklenince transkriptler değişti.")
    for first, last in zip(before, after):
        print(f"\n{first['id']}\nReferans: {first['reference']}\nÖnce: {first['hypothesis']}\nSonra: {last['hypothesis']}")
    print("PASS: Yeniden yüklenen projector seçilen transkriptleri tekrar üretti.", flush=True)


if __name__ == "__main__":
    main()
