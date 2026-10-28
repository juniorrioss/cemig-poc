# Resultados F0 — LFM3-1.7B bf16 baseline (benchmark_mobile, chunks_512)

## Célula: closed_book (n=1536)

| vocabulário | n | aprovação | cobertura | alucinação | recusa indevida | citação | truncadas | tokens méd. |
|---|---|---|---|---|---|---|---|---|
| all | 1536 | 5.9% | 0.318 | 45.5% | 57.5% | 85.2% | 2.1% | 1528 |
| clean | 384 | 7.0% | 0.249 | 25.0% | 55.7% | 87.8% | 1.6% | 1330 |
| technical | 384 | 8.6% | 0.298 | 22.7% | 59.1% | 87.5% | 1.3% | 1335 |
| operational | 384 | 4.2% | 0.419 | 65.6% | 57.6% | 79.2% | 4.7% | 1800 |
| lay | 384 | 3.6% | 0.306 | 68.8% | 57.6% | 86.2% | 1.0% | 1649 |

## Célula: oraculo (n=1536)

| vocabulário | n | aprovação | cobertura | alucinação | recusa indevida | citação | truncadas | tokens méd. |
|---|---|---|---|---|---|---|---|---|
| all | 1536 | 83.7% | 0.965 | 10.0% | 5.9% | 93.8% | 1.4% | 1056 |
| clean | 384 | 87.5% | 0.967 | 8.3% | 3.6% | 94.5% | 1.0% | 953 |
| technical | 384 | 84.1% | 0.967 | 10.4% | 4.7% | 95.1% | 1.0% | 1013 |
| operational | 384 | 81.8% | 0.969 | 10.7% | 7.0% | 93.8% | 1.0% | 1080 |
| lay | 384 | 81.2% | 0.955 | 10.4% | 8.1% | 91.9% | 2.3% | 1180 |

## Célula: ranking (n=1536)

| vocabulário | n | aprovação | cobertura | alucinação | recusa indevida | citação | truncadas | tokens méd. |
|---|---|---|---|---|---|---|---|---|
| all | 1536 | 65.8% | 0.852 | 18.2% | 6.2% | 87.0% | 10.4% | 1758 |
| clean | 384 | 70.6% | 0.857 | 13.3% | 7.8% | 87.8% | 10.4% | 1635 |
| technical | 384 | 73.2% | 0.885 | 13.5% | 5.2% | 90.6% | 7.8% | 1629 |
| operational | 384 | 62.0% | 0.836 | 21.1% | 5.5% | 85.7% | 12.0% | 1838 |
| lay | 384 | 57.6% | 0.831 | 25.0% | 6.2% | 84.1% | 11.5% | 1931 |

## Célula: distrator (n=1536)

| vocabulário | n | recusa correta (=aprov.) | alucinação | acertou mesmo assim | citação | truncadas | tokens méd. |
|---|---|---|---|---|---|---|---|
| all | 1536 | 30.5% | 28.3% | 20.2% | 60.3% | 35.4% | 2434 |
| clean | 384 | 39.6% | 16.1% | 12.5% | 61.7% | 33.6% | 2271 |
| technical | 384 | 43.2% | 21.9% | 16.4% | 66.7% | 30.2% | 2234 |
| operational | 384 | 18.5% | 33.1% | 26.6% | 51.3% | 43.2% | 2643 |
| lay | 384 | 20.8% | 42.2% | 25.5% | 61.5% | 34.4% | 2586 |

