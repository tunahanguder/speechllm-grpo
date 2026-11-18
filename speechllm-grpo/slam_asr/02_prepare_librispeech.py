"""Prepare the train-clean-100 manifest and a deterministic ASR pilot."""
import argparse
import json
import random
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

SPLIT = "train-clean-100"
MEMBER = re.compile(r"LibriSpeech/train-clean-100/(\d+)/(\d+)/([^/]+)")


def write_rows(path, rows):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False,
                                     encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    Path(stream.name).replace(path)


def prepare(root, seed, count):
    archive = root / "data" / "archives" / (SPLIT + ".tar.gz")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    audio = {}
    texts = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            match = MEMBER.fullmatch(member.name)
            if not match or not member.isfile():
                continue
            speaker, chapter, name = match.groups()
            prefix = f"{speaker}-{chapter}"
            if name == prefix + ".trans.txt":
                is_audio = False
            elif re.fullmatch(prefix + r"-\d+\.flac", name):
                is_audio = True
            else:
                continue
            path = root / "data" / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists() or path.stat().st_size != member.size:
                temp = path.with_suffix(path.suffix + ".part")
                with tar.extractfile(member) as source, temp.open("wb") as dest:
                    shutil.copyfileobj(source, dest)
                temp.replace(path)
            if is_audio:
                uid = path.stem
                if uid in audio:
                    raise ValueError(f"Duplicate audio: {uid}")
                audio[uid] = path
            else:
                for line in path.read_text(encoding="utf-8").splitlines():
                    uid, transcript = line.split(maxsplit=1)
                    if not uid.startswith(prefix + "-") or uid in texts:
                        raise ValueError(f"Invalid or duplicate transcript: {uid}")
                    texts[uid] = transcript
    unmatched = audio.keys() ^ texts.keys()
    if unmatched:
        raise ValueError(f"Unmatched audio/transcripts: {len(unmatched)}, e.g. {min(unmatched)}")
    if not 1 <= count <= len(audio):
        raise ValueError(f"Pilot size {count} incompatible with {len(audio)} pairs")
    rows = []
    for uid, path in sorted(audio.items()):
        # Native FLAC starts with STREAMINFO. Its packed field gives rate and sample count.
        with path.open("rb") as stream:
            header = stream.read(42)
        if len(header) != 42 or header[:4] != b"fLaC" or header[4] & 127 or int.from_bytes(header[5:8], "big") != 34:
            raise ValueError(f"Invalid FLAC STREAMINFO: {path}")
        packed = int.from_bytes(header[18:26], "big")
        samples = packed & ((1 << 36) - 1)
        if packed >> 44 != 16000 or samples == 0:
            raise ValueError(f"Unexpected sample rate or duration: {path}")
        rows.append(dict(id=uid, audio_path=str(path.relative_to(root)),
                         duration_seconds=samples / 16000, text=texts[uid]))
    out = root / "manifests"
    out.mkdir(exist_ok=True)
    manifest = out / (SPLIT + ".jsonl")
    pilot = out / f"pilot_{count}_seed{seed}.jsonl"
    write_rows(manifest, rows)
    write_rows(pilot, random.Random(seed).sample(rows, count))
    print(f"{len(rows)} pairs, {sum(r['duration_seconds'] for r in rows)/3600:.2f} hours")
    print(manifest, pilot, sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/slam_asr"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot-size", type=int, default=64)
    args = parser.parse_args()
    prepare(args.root, args.seed, args.pilot_size)
