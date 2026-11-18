"""YZM5110 – Vicuna ile tek batch eğitim kontrolü. Sürüm 1.0.1.

Girdi: Pilot manifestinin ilk iki ses–metin çifti.
Çıktı: Kayıp, gradyan normu, ağırlık değişimi ve GPU bellek ölçümü.
Kullanım: GPU oturumunda python 03_check_vicuna_batch.py. Checkpoint yazılmaz.
"""
import json
from pathlib import Path

import soundfile as sf
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, WhisperFeatureExtractor, WhisperModel

ROOT = Path("/content/drive/MyDrive/slam_asr")
WHISPER = "openai/whisper-small"
VICUNA = "lmsys/vicuna-7b-v1.5"


def main():
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU etkin bir Colab oturumu gerekiyor.")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("GPU:", torch.cuda.get_device_name(), "dtype:", dtype, flush=True)
    manifest = ROOT / "manifests/pilot_64_seed42.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()[:2]]
    assert len(rows) == 2
    waves = []
    for row in rows:
        wave, rate = sf.read(ROOT / row["audio_path"], dtype="float32")
        assert rate == 16000 and wave.ndim == 1 and len(wave) <= 30 * rate, row["id"]
        waves.append(wave)

    features = WhisperFeatureExtractor.from_pretrained(WHISPER)(
        waves, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device, dtype)
    encoder = WhisperModel.from_pretrained(WHISPER).encoder.to(device, dtype)
    encoder.eval().requires_grad_(False)
    # Encoder sabit ve önünde eğitilecek katman yok; aktivasyon gradyanlarını saklamak gerekmiyor.
    with torch.no_grad():
        frames = encoder(features).last_hidden_state
    batch, steps, width = frames.shape
    # Beş ardışık çerçeveyi özellik ekseninde birleştir: [B,T,H] → [B,T//5,5H].
    groups = frames[:, :steps // 5 * 5].reshape(batch, steps // 5, width * 5)
    tokenizer = AutoTokenizer.from_pretrained(VICUNA)
    llm = AutoModelForCausalLM.from_pretrained(VICUNA, torch_dtype=dtype).to(device)
    llm.eval().requires_grad_(False)
    embedding = llm.get_input_embeddings()
    projector = nn.Sequential(nn.Linear(width * 5, 2048), nn.ReLU(),
                              nn.Linear(2048, embedding.embedding_dim)).to(device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-4, weight_decay=0)
    # Projector, gruplanmış ses temsilini Vicuna token embedding boyutuna taşır.
    audio = projector(groups.float()).to(dtype)

    def tokens(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def embed(ids):
        return embedding(torch.tensor(ids, device=device))

    prefix = embed([tokenizer.bos_token_id] + tokens("USER: "))
    prompt = embed(tokens(" Transcribe speech to text. ASSISTANT:"))
    sequences, targets = [], []
    for sound, row in zip(audio, rows):
        ids = tokens(" " + row["text"]) + [tokenizer.eos_token_id]
        context = torch.cat([prefix, sound, prompt])
        sequences.append(torch.cat([context, embed(ids)]))
        # -100 etiketli ses ve komut konumları kayıp hesabından çıkarılır.
        # Hedef tokenlar ve bitiş tokenı denetimli öğrenme sinyalini sağlar.
        targets.append(torch.tensor([-100] * len(context) + ids, device=device))
    pad = nn.utils.rnn.pad_sequence
    inputs = pad(sequences, batch_first=True)
    labels = pad(targets, batch_first=True, padding_value=-100)
    lengths = torch.tensor([len(seq) for seq in sequences], device=device)
    mask = torch.arange(inputs.size(1), device=device)[None, :] < lengths[:, None]
    print("Encoder / grouped / projected:", frames.shape, groups.shape, audio.shape, flush=True)
    # Vicuna ağırlıkları sabit olsa da gradyanın girdiden projector’a ulaşması gerekir.
    # Bu yüzden bu ileri geçişte no_grad kullanılmaz.
    loss = llm(inputs_embeds=inputs, labels=labels, attention_mask=mask, use_cache=False).loss
    assert torch.isfinite(loss), "Non-finite loss"
    loss.backward()
    weight = projector[-1].weight
    grad = weight.grad.norm().item()
    assert 0 < grad < float("inf"), "Invalid projector gradient"
    assert all(p.grad is None for module in (encoder, llm) for p in module.parameters())
    # Adım öncesi kopyayla karşılaştırarak optimizer’ın ağırlığı değiştirdiğini doğrula.
    before = weight.detach().clone()
    optimizer.step()
    change = (weight.detach() - before).abs().max().item()
    assert 0 < change < float("inf"), "Invalid projector update"
    print(f"PASS | loss={loss.item():.4f} | grad_norm={grad:.6g} | max_delta={change:.6g}")
    print("Supervised tokens:", labels.ne(-100).sum(1).tolist())
    print(f"Peak GPU allocation: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
