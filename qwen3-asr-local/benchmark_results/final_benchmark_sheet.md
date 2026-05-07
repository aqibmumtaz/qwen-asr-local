# Final Full Benchmarking Sheet

## Scope

- ASR model: Qwen3-ASR-1.7B-Q8_0 + mmproj-Qwen3-ASR-1.7B-bf16
- Translation models: Qwen3-8B-Q5_K_M, Qwen3-8B-Q8_0, Qwen3-8B-Q4_K_M, Qwen3-4B-Q8_0, Qwen3-4B-Q4_K_M
- Columns include: Inference time (total, ASR, translation), Peak RAM, GPU usage, Transcripts, English & Urdu translations

## Audio Durations

| Audio | Duration (s) |
|---|---:|
| sample_en1.wav | 9.02 |
| sample_en2.wav | 9.00 |
| sample_ur1.wav | 6.00 |
| sample_ur2.wav | 5.98 |

## Model Summary (ASR + Translation)

| Model | Avg Total / file (s) | Avg ASR (s) | Avg Translation (s) | Avg sec/min audio | OK | Fail | Peak RAM (GB) | GPU Usage |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-4B-Q4_K_M | 49.50 | 20.00 | 29.50 | 411.94 | 4 | 0 | 10.36 | N/A (CPU-only, Metal OFF) |
| Qwen3-4B-Q8_0 | 58.50 | 20.00 | 38.50 | 492.83 | 4 | 0 | 10.37 | N/A (CPU-only, Metal OFF) |
| Qwen3-8B-Q4_K_M | 71.00 | 20.00 | 51.00 | 581.15 | 4 | 0 | 13.72 | N/A (CPU-only, Metal OFF) |
| Qwen3-8B-Q5_K_M | 514.00 | 20.00 | 494.00 | 4757.83 | 4 | 0 | 11.44 | N/A (CPU-only, Metal OFF) |
| Qwen3-8B-Q8_0 | 527.00 | 20.00 | 507.00 | 3643.50 | 4 | 0 | 14.10 | N/A (CPU-only, Metal OFF) |

## RAM / GPU Usage Summary

| Component | Peak RAM (GB) | GPU | Method |
|---|---:|---|---|
| ASR only | 10.38 | N/A (CPU-only, Metal OFF) | /usr/bin/time -l |
| ASR + Qwen3-8B-Q5_K_M | 11.44 | N/A (CPU-only, Metal OFF) | /usr/bin/time -l |
| ASR + Qwen3-8B-Q8_0 | 14.10 | N/A (CPU-only, Metal OFF) | /usr/bin/time -l |
| ASR + Qwen3-8B-Q4_K_M | 13.72 | N/A (CPU-only, Metal OFF) | /usr/bin/time -l |
| ASR + Qwen3-4B-Q8_0 | 10.37 | N/A (CPU-only, Metal OFF) | /usr/bin/time -l |
| ASR + Qwen3-4B-Q4_K_M | 10.36 | N/A (CPU-only, Metal OFF) | /usr/bin/time -l |

## Full Per-File Results

| Model | Audio | Total (s) | ASR (s) | Trans (s) | sec/min | Peak RAM (GB) | Status | Transcript | English | Urdu |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|
| Qwen3-8B-Q5_K_M | sample_en1.wav | 202.00 | 20.00 | 182.00 | 1343.09 | 11.44 | OK | Hello. This is a tes... | Hello. This is a tes... | ہیللو۔ یہ کوئنے کی ب... |
| Qwen3-8B-Q5_K_M | sample_en2.wav | 272.00 | 20.00 | 252.00 | 1813.33 | 11.44 | OK | Good morning. Today ... | Good morning. Today ... | صبح کی خوشیاں۔ آج ہم... |
| ⭐ Qwen3-8B-Q5_K_M | sample_ur1.wav | 215.00 | 20.00 | 195.00 | 2150.00 | 11.44 | OK | मेरा नाम अकीब है। ये... | My name is Akib. Thi... | میرا نام اکیب ہے۔ یہ... |
| ⭐ Qwen3-8B-Q5_K_M | sample_ur2.wav | 1367.00 | 20.00 | 1347.00 | 13724.90 | 11.44 | OK | आज का मौसम बहुत अच्छ... | Today's weather is v... | آج کا موسم بہت اچھا ... |
| Qwen3-8B-Q8_0 | sample_en1.wav | 1851.00 | 20.00 | 1831.00 | 12307.18 | 14.10 | OK | Hello. This is a tes... | Hello. This is a tes... | ہللو۔ یہ کوئنے کے بو... |
| Qwen3-8B-Q8_0 | sample_en2.wav | 92.00 | 20.00 | 72.00 | 613.33 | 14.10 | OK | Good morning. Today ... | Good morning. Today ... | صبح کے نامہ نگاری۔ آ... |
| ⭐ Qwen3-8B-Q8_0 | sample_ur1.wav | 78.00 | 20.00 | 58.00 | 780.00 | 14.10 | OK | मेरा नाम अकीब है। ये... | My name is Akib. Thi... | میرا نام اکیب ہے۔ یہ... |
| ⭐ Qwen3-8B-Q8_0 | sample_ur2.wav | 87.00 | 20.00 | 67.00 | 873.49 | 14.10 | OK | आज का मौसम बहुत अच्छ... | Today's weather is v... | آج کا موسم بہت اچھا ... |
| Qwen3-8B-Q4_K_M | sample_en1.wav | 74.00 | 20.00 | 54.00 | 492.02 | 13.72 | OK | Hello. This is a tes... | Hello. This is a tes... | ہیلو۔ یہ کوئنے کی بو... |
| Qwen3-8B-Q4_K_M | sample_en2.wav | 81.00 | 20.00 | 61.00 | 540.00 | 13.72 | OK | Good morning. Today ... | Good morning. Today ... | صبح کا خانہ ہے۔ آج ہ... |
| ⭐ Qwen3-8B-Q4_K_M | sample_ur1.wav | 65.00 | 20.00 | 45.00 | 650.00 | 13.72 | OK | मेरा नाम अकीब है। ये... | My name is Akib. Thi... | میرا نام اکیب ہے۔ یہ... |
| ⭐ Qwen3-8B-Q4_K_M | sample_ur2.wav | 64.00 | 20.00 | 44.00 | 642.57 | 13.72 | OK | आज का मौसम बहुत अच्छ... | Today's weather is v... | آج کا موسم بہت اچھا ... |
| Qwen3-4B-Q8_0 | sample_en1.wav | 59.00 | 20.00 | 39.00 | 392.29 | 10.37 | OK | Hello. This is a tes... | Hello. This is a tes... | سلام. ایک تجربے کا ا... |
| Qwen3-4B-Q8_0 | sample_en2.wav | 52.00 | 20.00 | 32.00 | 346.67 | 10.37 | OK | Good morning. Today ... | Good morning. Today ... | صبح کا سلام. آجہم او... |
| Qwen3-4B-Q8_0 | sample_ur1.wav | 64.00 | 20.00 | 44.00 | 640.00 | 10.37 | OK | मेरा नाम अकीब है। ये... | My name is Akib. Thi... | میرا نام اکیب ہے۔ یہ... |
| Qwen3-4B-Q8_0 | sample_ur2.wav | 59.00 | 20.00 | 39.00 | 592.37 | 10.37 | OK | आज का मौसम बहुत अच्छ... | Today's weather is v... | आज کا مسکین بہت اچھا... |
| Qwen3-4B-Q4_K_M | sample_en1.wav | 50.00 | 20.00 | 30.00 | 332.45 | 10.36 | OK | Hello. This is a tes... | Hello. This is a tes... | ہیلو ۔ یہ کوئنے کی س... |
| Qwen3-4B-Q4_K_M | sample_en2.wav | 50.00 | 20.00 | 30.00 | 333.33 | 10.36 | OK | Good morning. Today ... | Good morning. Today ... | صبح کا ہیلو، آجہم اپ... |
| Qwen3-4B-Q4_K_M | sample_ur1.wav | 49.00 | 20.00 | 29.00 | 490.00 | 10.36 | OK | मेरा नाम अकीब है। ये... | My name is Akib. Thi... | میرا نام اکیب ہے۔ یہ... |
| Qwen3-4B-Q4_K_M | sample_ur2.wav | 49.00 | 20.00 | 29.00 | 491.97 | 10.36 | OK | आज का मौसम बहुत अच्छ... | Today's weather is v... | आज کا میوسم بہت اچھا... |

## Notes

- **audio_seconds**: Duration of audio file
- **asr_seconds**: ASR-only inference time
- **translation_seconds_est**: Estimated = total - ASR
- **sec/min audio**: Normalized to 60 seconds of audio
- **asr_peak_ram_gb**: Peak RAM (GB) during ASR only
- **pipeline_peak_ram_gb**: Peak RAM (GB) during full pipeline (ASR + translation)
- **gpu_usage**: N/A because CPU-only mode (Metal OFF)
