# exp084_pf_confidence_residual_clip 結果

## 状態

Kaggle inference v1 と提出が完了。採用しない。

## 目的

`pf_confidence_residual_clip` を exp077 の付随変更ではなく、独立した backlog 実験として Kaggle inference で検証する。

## 設定

- parent: `exp077_full_replay_postprocess_guard`
- source model: `exp073_gpu_reproducibility_guard_for_exp063_full_replay`
- policy: `pf_confidence_residual_clip_q995`
- residual clip limit: `66.5908203125`

## 結果

- kernel: `kentookumura/exp084-pf-confidence-residual-clip-infer` version 1
- output: `/tmp/kaggle-output/exp084-pf-confidence-residual-clip-infer-v1`
- policy: `pf_confidence_residual_clip_q995`
- postprocess adjusted rows: `0`
- rows: `14,151`
- prediction range: `11593.671875` - `12241.693359375`
- submission SHA256: `7335854727543eff5db04873154394acae83274b18e73ed68d76491c4504788b`
- prediction SHA256: `2e47e986c013acfafaa01c652d47649778db5616e40dc3130e4e12dede7b7502`
- test feature content SHA256: `e3567a64807a16c3c4d80fe6bca2611ba3fe8d13b4b20be4540e8d1ac354965c`
- submit-check: PASS
- submitted refs: `53854829`, `53854846`
- Public LB: `8.746`

public sample の 3 wells では、raw residual が dynamic clip limit 内に収まっていたため no-op だった。local submission SHA は exp073 deterministic output と同じで、Public LB `8.746` は exp077 `8.611` より悪い。したがって exp084 は採用しない。
