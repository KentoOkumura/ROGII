# exp508_exp413_public_trajectory_postprocess_audit

## 状態

- ルート: `ml_model`
- 状態: Kaggle Stage A version 1完了、promotion FAILで終端閉鎖
- CV: `7.878669066831366`（親exp413 `7.884802794404715`から`0.006133728 ft`改善）
- Public LB: なし（親exp413参照は`7.201`）
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-08-04
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`
- public-core参照: `exp497_strict_public_core_fold_safe_ensemble_on_exp413`

## 仮説

exp413の最終TVT予測に、公開ノートブックと同じwell別Savitzky--Golay
`window=61 / polyorder=3`を一度だけ適用すると、upstream modelやphysics componentを
変更せず、高周波なtrajectory揺れだけを減らして小さいが安定した改善を得られる。

## 固定した変更点

- selectable primaryは`sg61_p3_final_tvt`の1本だけ。
- controlは保存済みexp413 Stage D OOF。
- `tau=85` warmup単独とwarmup+SGはreport-onlyで、昇格・救済不可。
- 公開のmodel 60% / direct LikPF 40% blendは使わない。
- SG後のreanchor、clip、projection、window/tau gridは行わない。
- model、booster、HMM、PF、Beam、親予測を再学習・再生成しない。

## 検証方針

- Fold: exp413と同一outer 5 folds
- Group: `well`
- Metric: suffix-row unweighted RMSE
- Leakage check: input / key / row order / foldと全候補predictionをtruth接続前にSHA freeze
- Primary gate: gain`>=0.01 ft`、4/5 folds、固定5 scope`<=+0.02 ft`、by-well p95/worst`<=+0.25 ft`、prediction-start continuityの全AND
- FAIL時: SG/tau/router/gateをsame-OOF救済せず終端閉鎖

公開source自身がSG効果を約`0.01 ft`と記述し、今回はmodel fitやparameter探索のない固定変換で
あるため、最小gainを`0.01 ft`とした。その代わりfold、scope、well-tail、開始境界を厳格に守る。

## well-level routing

exp508には含めない。exp508 primaryが全AND gateをPASSし、raw/SGのtarget-free disagreementに
独立な相補性が確認できた場合だけ、別実験の`requirements.md`・別承認で検討する。公開の固定
`n_eval / z_span` threshold、variant map、well ID、public cardinalityは再利用しない。

## 実行量

Stage Aは保存OOFだけの1 primary + 2 report-only、0 model / 0 booster /
0 HMM/PF/Beam / 0 GPUでKaggle private CPU version 1を完了した。正規train Notebookは採用済み。
推論、提出は未実装・未実行である。

## 実装

- `exp508_exp413_public_trajectory_postprocess_audit_compact_selfcontained_train.py`:
  notebook-safeなJupytext percent形式のStage A候補。
- `exp508_exp413_public_trajectory_postprocess_audit_compact_selfcontained_train.ipynb`:
  上記sourceから生成した別名Notebook候補。
- `exp508_exp413_public_trajectory_postprocess_audit_train.ipynb`:
  ユーザー承認後に採用し、Kaggle Stage A version 1で実行した正規train Notebook。
- `test_exp508_contract.py`: SG同値性、short-well、truth-free freeze、truth-late join、
  report-only分離、all-AND gate、run承認guardを検証する。

## 所見

SG61/p3は`7.884802794 → 7.878669067`、5/5 folds、固定5 scope、well-tail、開始境界を
一貫して改善・保護し、trajectory second-difference RMSも`0.530398 → 0.011205 ft`へ低下した。
ただしpooled gainは`0.006133728 ft`で、事前固定した最低`0.01 ft`に届かなかった。
結果後のgate緩和やreport-only救済をせず、branchを閉じる。

## 参照ファイル

- `config.yaml`: 入力SHA、候補、gate、実行量、後続router条件
- `postprocess_contract.yaml`: SG/warmupの機械可読契約
- `output_contract.md`: 将来保存する生成物
- `SESSION_NOTES.md`: 設計・実装・Kaggle実行・終端判断の記録
- `result.md`: Stage A実測値とpromotion gate
- steering: `../../docs/legacy/steering/20260804-exp508-exp413-public-trajectory-postprocess-audit/`

## 次

exp508は終端閉鎖した。SG/tau/routerのsame-OOF救済、推論、提出へ進まない。
