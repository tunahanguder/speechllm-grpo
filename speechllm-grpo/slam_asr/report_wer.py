"""Kaydedilmiş değerlendirme CSV'lerinden WER tablosunu üretir."""
import csv
import math
from pathlib import Path

print("| Koşu | Kontrol noktası | Küme | WER (%) |")
print("|---|---|---|---:|")
for path in sorted(Path(__file__).with_name("results").glob("*.csv")):
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            errors = sum(int(row[key]) for key in ("S", "D", "I"))
            wer = 100 * errors / int(row["N"])
            if not math.isclose(wer, float(row["wer_percent"]), abs_tol=0.011):
                raise ValueError(f"WER uyuşmazlığı: {path.name} {row}")
            print(f"| {row['run']} | {row['checkpoint']} | {row['split']} | {wer:.2f} |")
