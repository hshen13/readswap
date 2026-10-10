# Data

## Benchmark files (`confirmatory/`)

The five confirmatory benchmark files are copied unchanged from the LaSeR repository
(https://github.com/RUCBM/LaSeR, commit `48996f78e76986982f8678a892c21483998823fe`, folder `data/`).
They are used for evaluation only.

| File | Problems | SHA-256 | Original source and license |
|---|---:|---|---|
| `confirmatory/math500/test.jsonl` | 500 | `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` | MATH500 split of PRM800K (MIT), a subset of MATH (MIT) |
| `confirmatory/amc23/test.jsonl` | 40 | `9e2f9d5c955b2eb00d34170e6eb805e67d7ceea174006a30f993c2a40511d4ab` | AMC 2023 competition problems; no license stated in the LaSeR repository, see the original source |
| `confirmatory/aime24/test.jsonl` | 30 | `1c1004689dbaec0bd43b71f7083cabffefc27716783e19235947dc355031a95c` | AIME 2024 competition problems; no license stated in the LaSeR repository, see the original source |
| `confirmatory/aime25/test.jsonl` | 30 | `b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331` | AIME 2025 competition problems; no license stated in the LaSeR repository, see the original source |
| `confirmatory/OlympiadBench/test.jsonl` | 675 | `5ee85c9bdfd623654b4172c7b3168478ff5b57d3e45b7d3e1f2089178f5d2728` | OlympiadBench (MIT), the subset shipped with LaSeR |

The LaSeR repository itself is released under the MIT License (Copyright (c) 2025 RUCBM).
AMC and AIME problems are competition problems of the Mathematical Association of America.

## Pilot file (`pilot/`)

`pilot/DeepMath-103K_train_seed0_n40.jsonl` holds 40 problems drawn with seed 0 from the training split of
DeepMath-103K (https://huggingface.co/datasets/zwhe99/DeepMath-103K, MIT License, revision
`5cf055d1fe3d7a2eb19719ac020211469736ae44`), after removing every problem whose text equals a confirmatory problem.
It is used only by the pilot (`code/run_e1_rest.sh`) and enters no table of the paper.

## Registries

- `LASER_CONFIGS.json`: per checkpoint, the Hugging Face id and revision, the marker token and its id, the
  reference constant, the ratio coefficient, the three R1 tokens, the prompt template, the generation and readout
  decoding, and the base model used by R5. Entries of the form `LaSeR@48996f7:<file>:<lines>` point to the file
  and lines of the LaSeR repository at that commit from which a setting was read. `local_path` is relative to the
  repository root (`data/models/<key>`); a null `base_model_local_path` is resolved at run time to the
  snapshot of `base_model_repo_id` at `base_model_revision` in the Hugging Face cache (`HF_HOME`).
- `MODELS.json`: the same checkpoints and base models in list form.
- `MANIFEST.json`: provenance, row counts and SHA-256 hashes of the benchmark and pilot files, and of the
  DeepMath-103K parquet shards from which the pilot was drawn (the shards themselves are not included).

Model weights (`data/models/`) are not included; see the top-level README for the download commands.
