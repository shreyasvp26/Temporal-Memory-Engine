# LongMemEval Benchmark Report

- **System:** `vector_only`
- **Baseline:** `None`
- **Total Questions:** `30`
- **Overall Accuracy:** `96.7%`
- **Overall Token F1:** `32.0%`
- **Abstention Accuracy:** `100.0%`

## Category Breakdown

| Category | N | Accuracy | 95% CI | Token F1 | Exact Match |
| :--- | :---: | :---: | :---: | :---: | :---: |
| abstention | 6 | 100.0% | [100.0%, 100.0%] | 96.7% | 83.3% |
| knowledge-update | 6 | 83.3% | [50.0%, 100.0%] | 10.3% | 0.0% |
| multi-session | 6 | 100.0% | [100.0%, 100.0%] | 13.0% | 0.0% |
| single-session-user | 6 | 100.0% | [100.0%, 100.0%] | 27.7% | 0.0% |
| temporal-reasoning | 6 | 100.0% | [100.0%, 100.0%] | 12.4% | 0.0% |
| **Overall** | **30** | **96.7%** | **[90.0%, 100.0%]** | **32.0%** | **16.7%** |

## Latency Profile

| Metric | Latency (ms) |
| :--- | :---: |
| p50 | 5310.1 |
| p95 | 6634.9 |
| p99 | 7083.2 |
| Mean | 4878.2 |
