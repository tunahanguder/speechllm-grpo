# SLAM-ASR ve GRPO ile Otomatik Konuşma Tanıma

**Tunahan Güder — YZM5110 Çok Modlu Öğrenme Araştırma Projesi**

Bu projede Whisper-small ses kodlayıcısı, eğitilebilir bir projektör
üzerinden Vicuna veya Qwen dil modeline bağlanır. LibriSpeech üzerinde
dil modeli seçiminin, projektör aktivasyonunun ve GRPO ödül tasarımının
kelime hata oranına (WER) etkisi incelenir.

## Başlangıç

Ana çalıştırma dosyası [slam_asr_github.ipynb](slam_asr_github.ipynb)
notebook’udur. Notebook’u Google Colab’da açın. Eğitim ve model
değerlendirmesi için CUDA ve BF16 destekli bir GPU çalışma zamanı seçin.

1. Drive’ı bağlayın ve `PROJECT_ROOT` değerini proje klasörünü gösterecek
   şekilde ayarlayın. Python betikleri ve `requirements.txt` bu dizinde
   bulunmalıdır.
2. Proje dizininden `python -m pip install -r requirements.txt` komutuyla
   bağımlılıkları kurun. PyTorch için Colab’ın CUDA ortamıyla uyumlu bir
   sürüm kullanın.
3. [LibriSpeech](https://www.openslr.org/12/) arşivlerini özgün dosya
   adlarıyla `data/archives/` altına koyun. Eğitim için `train-clean-100`,
   `train-clean-360` ve `train-other-500`, değerlendirme için `dev-other`,
   `test-clean` ve `test-other` kullanılır.
4. Eğitimden önce `06_prepare_train960.py` ile `manifests/train-960.jsonl`
   dosyasını oluşturun. Ardından notebook’ta seçtiğiniz deneyin eğitim
   veya değerlendirme hücresini çalıştırın.

## Deneyler

| Deney | Model veya müdahale | Eğitim başlangıcı |
|---|---|---|
| A | Whisper-small + Vicuna-7B-v1.5. Yalnızca projektör eğitilir. | Yeni projektör |
| B | Whisper-small + Qwen3.5-0.8B Base. ReLU içeren projektör eğitilir. | Yeni projektör |
| B′ | B deneyindeki projektörde ReLU yerine Identity kullanılır. | Yeni projektör |
| E | Whisper-small + Qwen3.5-9B Base. Yalnızca projektör eğitilir. | Yeni projektör |
| C | Negatif WER ödülüyle GRPO. | B-best |
| D | Tam eşleşme ve parçalı WER ödülüyle GRPO. | B-best |
| F-0,05 | D ödülüne tek kelime hatası için 0,05 bonus eklenir. | B-best |
| F-0,10 | D ödülüne tek kelime hatası için 0,10 bonus eklenir. | B-best |

C, D ve F eğitimlerinde ses kodlayıcısı ve projektör sabit tutulur,
dil modelinin LoRA parametreleri güncellenir. İki F koşusu ayrı oturumlarda
çalıştırılabilir. Her biri kendi koşu klasörüne yazar.

## Değerlendirme ve sonuçlar

Ana test sonuçları `test-clean` ve `test-other` üzerinde greedy decoding
(`beam=1`) ile elde edilmiştir. A ve B için `beam=4` ayrı bir ek
değerlendirmedir. F koşuları, D’nin 810. adımındaki kontrol noktasıyla
karşılaştırılır.

Notebook’taki değerlendirme hücreleri, ilgili kontrol noktaları hazırsa
yeniden eğitim yapmadan çalıştırılabilir. Değerlendirme çıktıları `runs/`
altına yazılır. `comparison.csv` toplu ölçümleri, `predictions.jsonl` ise
kayıt düzeyindeki tahminleri içerir.

Depodaki kayıtlı CSV’lerden WER tablosunu yeniden hesaplamak için:

```bash
python speechllm-grpo/slam_asr/report_wer.py
```

Bu kontrol GPU, ses arşivleri veya model ağırlıkları gerektirmez.
Eğitim ve çıkarım için veri arşivleri ve ilgili ağırlıklar ayrıca gerekir.
