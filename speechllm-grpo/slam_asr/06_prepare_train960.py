"""YZM5110 • Sürüm 1.0.0 • LibriSpeech-960 eğitim manifesti.
Girdi: data/archives altında üç resmî tar.gz arşivi. GPU veya ek paket gerekmez.
Çıktı: manifests/train-960.jsonl ve train-960.summary.json.
İlk sürüm: Arşiv içinden başlık ve metin okur; FLAC dosyalarını Drive'a kopyalamaz.
"""
import argparse
import json
from pathlib import Path
import re
import tarfile
import tempfile

SPLITS = ("train-clean-100", "train-clean-360", "train-other-500")


def scan(archive, split, progress):
    sounds, texts = {}, {}
    prefix = f"LibriSpeech/{split}/"
    with tarfile.open(archive, "r|gz") as tar:
        for item in tar:
            if not item.isfile() or not item.name.startswith(prefix):
                continue
            path = Path(item.name)
            chapter = path.parent.name
            speaker = path.parent.parent.name
            pattern = re.escape(f"{speaker}-{chapter}-") + r"\d+"
            if path.suffix == ".flac":
                uid = path.stem
                if uid in sounds or not re.fullmatch(pattern, uid):
                    raise ValueError(f"Geçersiz veya yinelenen ses: {item.name}")
                # STREAMINFO içindeki toplam örnek sayısı / örnekleme hızı = süre.
                with tar.extractfile(item) as f:
                    header = f.read(42)
                if len(header)!=42 or header[:4]!=b"fLaC" or header[4]&127 or int.from_bytes(header[5:8],"big")!=34:
                    raise ValueError(f"FLAC başlığı geçersiz: {uid}")
                packed = int.from_bytes(header[18:26], "big")
                samples = packed & ((1 << 36)-1)
                if packed >> 44 != 16000 or ((packed >> 41) & 7) or not samples:
                    raise ValueError(f"16 kHz mono ses bekleniyor: {uid}")
                sounds[uid] = (item.name, samples)
                if len(sounds) % progress == 0:
                    print(f"{split}: {len(sounds):,} ses tarandı", flush=True)
            elif path.name.endswith(".trans.txt"):
                with tar.extractfile(item) as f:
                    for line in f:
                        uid, text = line.decode().strip().split(maxsplit=1)
                        if uid in texts or not re.fullmatch(pattern, uid):
                            raise ValueError(f"Geçersiz veya yinelenen transkript: {uid}")
                        texts[uid] = text
    mismatch = sounds.keys() ^ texts.keys()
    if mismatch or not sounds:
        raise ValueError(f"{split}: eşleşmeyen {len(mismatch)} kayıt veya boş arşiv")
    for uid in sorted(sounds):
        member, samples = sounds[uid]
        yield dict(id=uid, split=split, archive=f"data/archives/{archive.name}",
                   member=member, audio_path=f"data/{member}", duration_seconds=samples/16000,
                   text=texts[uid])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/slam_asr"))
    parser.add_argument("--progress-every", type=int, default=10000)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every pozitif olmalı.")
    out = args.root / "manifests"
    out.mkdir(parents=True, exist_ok=True)
    stats, seen = {}, set()
    # Her tamamlanan bölüm ayrı saklanır: Colab kesilirse tamamlanan arşiv tekrar taranmaz.
    for split in SPLITS:
        archive = args.root / "data/archives" / f"{split}.tar.gz"
        stat = archive.stat()
        cache = out / f"06_{split}_{stat.st_size}_{stat.st_mtime_ns}_v1.jsonl"
        if not cache.exists():
            print(f"Taranıyor: {split}", flush=True)
            with tempfile.NamedTemporaryFile("w", dir=out, delete=False, encoding="utf-8") as f:
                for row in scan(archive, split, args.progress_every):
                    f.write(json.dumps(row, ensure_ascii=False)+"\n")
            Path(f.name).replace(cache)
        stats[split] = {"cache":cache.name}
    # Üç bölüm tamamlanmadan nihai manifest yayımlanmaz.
    with tempfile.NamedTemporaryFile("w",dir=out,delete=False,encoding="utf-8") as merged:
        for split, info in stats.items():
            count, seconds, long_count = 0, 0.0, 0
            with (out/info["cache"]).open(encoding="utf-8") as src:
                for line in src:
                    row = json.loads(line)
                    if row["id"] in seen:
                        raise ValueError(f"Tekrarlanan kayıt: {row['id']}")
                    seen.add(row["id"])
                    merged.write(line)
                    count += 1
                    seconds += row["duration_seconds"]
                    long_count += row["duration_seconds"] > 30
            info.update(records=count, hours=seconds/3600, over_30s=long_count)
            print(f"{split}: {count:,} kayıt, {seconds/3600:.2f} saat, 30 sn üstü {long_count}")
    dest = out / "train-960.jsonl"
    Path(merged.name).replace(dest)
    report = dict(version="1.0.0", splits=stats, records=len(seen), hours=sum(i["hours"] for i in stats.values()))
    (out/"train-960.summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"TAMAM: {len(seen):,} kayıt, {report['hours']:.2f} saat. {dest}")


if __name__ == "__main__":
    main()
