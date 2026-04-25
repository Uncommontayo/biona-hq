# Biona — Biona Speech Intelligence Engine

## Repository Structure
- `axon/`   — Biona Axon: On-device C++17 inference engine
- `lab/` — Server-side Python training pipeline (Biona Lab)

## Biona Boundary Rule
**Biona Axon runs on-device. Biona Lab runs on servers. These are separate systems.**
LLM inference, cloud APIs, and large-batch GPU processing belong in Biona Lab ONLY.
The ONNX model produced by Biona Lab is the only artifact that crosses the boundary.
