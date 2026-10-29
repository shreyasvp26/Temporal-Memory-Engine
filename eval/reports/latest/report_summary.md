# LongMemEval Live Evaluation — Baseline Comparison

- **Dataset:** `data/longmemeval_stratified_30.json`
- **Model:** `gemini-3.5-flash-lite`
- **Questions:** `30`

| Baseline | Accuracy | Token F1 | Abstention Acc | p50 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: |
| vector_only | 96.7% | 32.0% | 100.0% | 5310 |
| bm25_only | 100.0% | 32.1% | 100.0% | 3817 |
| full_bitemporal | 96.7% | 33.2% | 100.0% | 4609 |
