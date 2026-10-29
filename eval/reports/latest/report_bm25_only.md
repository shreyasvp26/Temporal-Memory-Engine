# LongMemEval Benchmark Report

- **System:** `bm25_only`
- **Baseline:** `None`
- **Total Questions:** `30`
- **Overall Accuracy:** `100.0%`
- **Overall Token F1:** `32.1%`
- **Abstention Accuracy:** `100.0%`

## Category Breakdown

| Category | N | Accuracy | 95% CI | Token F1 | Exact Match |
| :--- | :---: | :---: | :---: | :---: | :---: |
| abstention | 6 | 100.0% | [100.0%, 100.0%] | 100.0% | 100.0% |
| knowledge-update | 6 | 100.0% | [100.0%, 100.0%] | 12.7% | 0.0% |
| multi-session | 6 | 100.0% | [100.0%, 100.0%] | 12.8% | 0.0% |
| single-session-user | 6 | 100.0% | [100.0%, 100.0%] | 22.7% | 0.0% |
| temporal-reasoning | 6 | 100.0% | [100.0%, 100.0%] | 12.2% | 0.0% |
| **Overall** | **30** | **100.0%** | **[100.0%, 100.0%]** | **32.1%** | **20.0%** |

## Latency Profile

| Metric | Latency (ms) |
| :--- | :---: |
| p50 | 3816.7 |
| p95 | 4601.9 |
| p99 | 4973.5 |
| Mean | 3932.0 |
