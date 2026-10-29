# LongMemEval Benchmark Report

- **System:** `full_bitemporal`
- **Baseline:** `None`
- **Total Questions:** `30`
- **Overall Accuracy:** `96.7%`
- **Overall Token F1:** `33.2%`
- **Abstention Accuracy:** `100.0%`

## Category Breakdown

| Category | N | Accuracy | 95% CI | Token F1 | Exact Match |
| :--- | :---: | :---: | :---: | :---: | :---: |
| abstention | 6 | 100.0% | [100.0%, 100.0%] | 100.0% | 100.0% |
| knowledge-update | 6 | 83.3% | [50.0%, 100.0%] | 16.7% | 0.0% |
| multi-session | 6 | 100.0% | [100.0%, 100.0%] | 10.5% | 0.0% |
| single-session-user | 6 | 100.0% | [100.0%, 100.0%] | 25.9% | 0.0% |
| temporal-reasoning | 6 | 100.0% | [100.0%, 100.0%] | 12.9% | 0.0% |
| **Overall** | **30** | **96.7%** | **[90.0%, 100.0%]** | **33.2%** | **20.0%** |

## Latency Profile

| Metric | Latency (ms) |
| :--- | :---: |
| p50 | 4609.2 |
| p95 | 5638.9 |
| p99 | 23654.4 |
| Mean | 5512.1 |
