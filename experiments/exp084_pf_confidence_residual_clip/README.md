# exp084_pf_confidence_residual_clip

## 状態

Kaggle inference v1 と提出が完了。`exp077_full_replay_postprocess_guard` は `longtail_likpf_tiny_gate_w006` の submitted anchor として保持し、この実験では `pf_confidence_residual_clip_q995` だけを分離して実行した。

## 仮説

exp073 deterministic full replay ML anchor の residual を、PF/Beam confidence に応じて clip すれば、hard switch や direct PF replacement より保守的に hidden transfer を試せる。

## 検証方針

exp073 saved booster inference と raw test PF/Beam/likelihood-PF feature replay を使う。新しい教師あり学習は行わず、`inference.postprocess_policy=pf_confidence_residual_clip_q995` の固定 policy として Kaggle inference を実行し、生成物 SHA、submit-check、必要なら Public LB を記録する。

## 所見

public sample の 3 wells / 14,151 rows では `postprocess_adjusted_rows=0` で no-op だった。`submission.csv` は submit-check PASS。duplicate submissions ref `53854829` / `53854846` は Public LB `8.746`。exp077 `8.611` より悪く、local output は exp073 no-op と同じ SHA のため採用しない。
