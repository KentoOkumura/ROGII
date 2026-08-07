# 要件

## 依頼

exp410で確認したacross-seed算術平均のoffset増幅に対し、単純平均を固定
likelihood-temperature-5重み付き平均へ置き換える案を独立実験として設計する。
2026-07-27の初期依頼ではbacklog、steering、実験scaffoldと設計確定までとした。
2026-07-28の`exp417を実装してください`により、Stage Aのcompact self-contained
候補とcontract testsまでを追加承認範囲とした。続く`実行してください`により、
正規train Notebook採用、Kaggle package / push、Stage A実行までを承認範囲に追加した。
推論と提出は承認しない。

## 制約

- Route: `pf_beam`
- artifact parent: `exp404_scale5_sigma_gr_likelihood_pf_ablation`
- scientific control: `exp072_exp063_full_replay_feature_cache`
- Stage Aはexp404 x1.0の同一PF seed bankから保存された
  `likpf_mean_x1p0`と`likpf_scale_5_x1p0`だけを比較する。
- temperatureは`5.0`に固定し、temperature grid、scale 3/8/12、best seed、
  median、mode、medoid、selectorを追加しない。
- Stage AのPF / model / booster / GPU実行はすべて0。
- GR sigma、roughening、process noise、ESS threshold、particle / seed数を変更しない。
- prediction identity / SHAをfreezeした後だけtruth、fold、hidden-like roleを読む。
- full-suffix GR likelihoodはsuffix TVTを使わず、Kaggle batch inferenceで全GRが
  利用可能な前提に限定する。causal online予測とは呼ばない。
- exp413のML置換・再学習とは独立に判定し、結果を再分類しない。
- Stage A PASS後もraw-test inference実装と実行には別承認が必要。
- 再現性は`docs/06_reproducibility.md`に従う。

## 受け入れ基準

- exp404 frozen predictionのraw/decompressed/logical/schema/contract SHAが一致する。
- 3,783,989 rows / 773 wells / folds 0--4でcontrol / candidateが有限かつID一対一。
- controlとcandidateが同じx1.0 particles / seeds / trajectory bank由来である。
- control RMSEがexp072に`1e-5 ft`以内で一致する。
- Stage A PF well-runs、model configs、trained folds、boosters、GPU runsが0。
- pooled RMSEを`0.05 ft`以上改善し、4/5 folds以上で改善する。
- raw-GR observedは`0.05 ft`以上改善し、raw-GR missing、high-missing、
  1000+、hidden-like 2面、by-well p95を悪化させない。
- worst-well regressionを`0.25 ft`以内、fixed HMM/LikPF 50:50を非悪化とする。
- scientific guardはAND条件。FAIL時はtemperatureやselectorを救済探索せず閉じる。
- PASSしてもinference候補化は同じexp417内の別設計・別承認とする。
- deterministic anchorとする場合はprediction/input/code/config/kernel/submission SHAと
  rerun parityを記録する。
