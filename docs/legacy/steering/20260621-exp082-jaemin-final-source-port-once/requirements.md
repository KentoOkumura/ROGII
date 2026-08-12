# 要件

## 依頼

`jaemin_final_source_port_once` を `exp082_public_artifact_replay_followup` に実装する。`jaemin3404/rogii-sp45-fleongg-blend-v2` の archived source から SP45 branch と fleongg pretrained branch を hidden test 上で再生成し、public output CSV copy ではなく source-port notebook として 1 回だけ提出候補を作る。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 新しい exp は切らず、既存 `exp082_public_artifact_replay_followup` に inference notebook、記録、metrics を追記する。
- `/kaggle/input/notebooks/...` や public notebook output の `submission.csv` を読まない。
- direct CSV submit ではなく Kaggle Notebook version からの submit を前提にする。
- Pilkwang raw / w0.60 branch は exact archived source missing のため混ぜない。

## 受け入れ基準

- `exp082_public_artifact_replay_followup_inference.ipynb` が jaemin source-port 版として読めるセル構成になっている。
- `validate_experiment` と Kaggle inference notebook prepare が pass する。
- Kaggle commit output の `submission.csv`、`sp45_projection_submission.csv`、`fleongg_pretrained_submission.csv`、`sp45_fleongg_blend_report.csv` が生成される。
- `submit-check` と `scripts/validate_submission.py` が pass する。
- `submission.csv` の rows、id order、SHA、prediction range、sidecar diff、Kaggle kernel version を `SESSION_NOTES.md` / `metrics.json` に記録する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。今回は public replay ensemble candidate として扱い、ML / PF 単独 anchor と混同しない。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
