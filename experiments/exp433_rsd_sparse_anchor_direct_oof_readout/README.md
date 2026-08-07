# exp433_rsd_sparse_anchor_direct_oof_readout

## 状態

- ルート: `pf_beam`
- 状態: `completed_scientific_failed_closed`
- 優先度: P2
- CV: `9.692148252`
- 基準exp226 CV: `9.427109597`
- Public / Private LB: - / -
- 作成日 / 完了日: 2026-07-28 / 2026-07-28
- 親:
  `exp426_rsd_binned_pattern_absolute_reanchor`
- base prediction:
  `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`

## 仮説

exp426のRSD-binned GR scoreは全blockへ密には供給できなかったが、
unsupported blockをcarryする固定Viterbiなら、疎なabsolute anchorでも
exp226の累積offsetを改善できる可能性がある。

## 固定した処理

- exp426 version 1 score:
  101,231 rows / 7,787 blocks / 13 offsets
- primary:
  initial sigma 5 ft、transition sigma 10 ft、
  first step 20 ft、adjacent step 40 ftの固定Viterbi
- unsupported block:
  全state emission 0のtransition-only carry
- row correction:
  suffix boundary 0と512-row block centerの線形補間、最後だけhold
- blockwise top-1:
  report-only
- truth:
  input / support / datum / prediction / independent rerun SHA freeze後だけjoin

score、support、rank、top-3、offset grid、decoder、transition、clip、
activation、blendは結果を見て変更していない。

## 検証方針

- primary:
  exp226保存OOF 3,783,989行に固定Viterbi correctionを直接加えたpooled RMSE
- reporting:
  5 folds、distance、raw-GR、hidden-like、persistent episode、by-well tail
- technical:
  入力SHA / inventory、parent RMSE、truth-late ledger、correction slope、
  independent full / probe rerun、runtime / memory
- scientific:
  pooled gain、4/5 folds、1000+、episode SSE / wells、guarded scopes、
  new episode SSE、by-well p95 / worstのAND gate
- coverage:
  必須reportだがtruth readやprimary activationのgateには使わない

## Kaggle実行

- kernel:
  `kentookumura/exp433-rsd-sparse-anchor-direct-oof-readout-train`
- version / id_no:
  `3 / 128939253`
- private CPU / internet off
- runtime / peak RSS:
  `122.701148 sec / 2.708778 GB`
- 実行量:
  1 decoder / 1 diagnostic / 773 wells / 5 folds
- model / booster / HMM / PF / Beam / GPU / parent再生成:
  すべて0

version 1 / 2のtechnical errorは、それぞれ非round-trip-safeなproducer SHA照合と
`fold` scope routingだった。固定科学契約を変えず回帰testを追加し、
version 3でtechnical gateを全PASSした。

## 結果

| メトリック | 値 |
| --- | ---: |
| exp433 direct OOF RMSE | 9.692148252 |
| exp226 base RMSE | 9.427109597 |
| gain | -0.265038655 ft |
| improvement folds | 0 / 5 |
| distance 1000+ gain | -0.298535385 ft |
| persistent episode SSE reduction | -2.797279% |
| persistent wells improved | 160 / 449 |
| by-well delta p95 | +3.282839 ft |
| worst-well regression | +15.926322 ft |

near 0--500 ftでは最大`+0.026029 ft`の小改善があったが、500+、
raw-GR missing、hidden-like、persistent episodeを悪化させた。
technical gateはPASS、scientific gateは全9条件FAIL。

## 再現性

- prediction SHA:
  `c461a14708ffc951060a77e0016a7947f7e2cae1abeb28b539465c0289100377`
- datum path SHA:
  `e3b4f9afbe0f431c5f80add93f11abb15af44dbae64fd9511be579e2d8bef96e`
- fixed probe SHA:
  `639fb28ff2397123b24d44fe3aaaa56570aa0840412f541072260a9f7af46b9a`
- independent full / probe rerun:
  一致

## ファイル

- Jupytext source:
  `exp433_rsd_sparse_anchor_direct_oof_readout_compact_selfcontained_train.py`
- canonical train Notebook:
  `exp433_rsd_sparse_anchor_direct_oof_readout_train.ipynb`
- compact candidate history:
  `exp433_rsd_sparse_anchor_direct_oof_readout_compact_selfcontained_train.ipynb`
- inference Notebook:
  placeholderのまま
- 詳細:
  `SESSION_NOTES.md`、`result.md`、`metrics.json`

## 結論

疎なRSD absolute-datum scoreを固定carry Viterbiで直接使う枝は、
実OOFで累積offsetを改善しなかった。同じscore family内のparameter救済、
activation、clip、blend、well gateは行わずterminal closeする。
inference / submissionは無効のまま。

## 所見

0--500 ftでは小さく改善した一方、疎なanchor間のcarryが500+で誤差を蓄積し、
pooled、全fold、persistent episode、hidden-like、by-well tailを悪化させた。
coverage不足だけでなく、supported scoreの方向性もabsolute datumとして弱い。
exp426 / exp433のRSD sparse-anchor familyは閉じ、既存P1を優先する。
