# 要件

## 依頼

- exp226 の最終 `tvt_pred` を行方向に差分化し、その増分を HMM の遷移中心へ毎行そのまま与える実験を設計する。
- まず HMM 版だけを対象とし、その結果をレビューしてから PF 版へ進む。
- 2026-07-30 の追加依頼「exp491を実装してください」に基づき、設計済みの
  fixed32 Stage 0 だけを compact self-contained notebook として実装する。
- Kaggle package、Stage 0 実行、Stage 1、inference、PF、提出は行わない。

## 実験の問い

exp226 の geometry-only 中間予測 `tvt_geop` より、最終 `tvt_pred` の行間増分の方が未知 suffix の局所 TVT rate をよく表現し、exp209 系 HMM の rate 追従遅れを軽減できるか。

## 制約

- 対象実験: `exp491_exp226_final_tvt_rate_direct_hmm`
- Route: `pf_beam`
- 親実験: `exp437_neighbor_geometry_tvt_only_transition_hmm`
- 予測ソース: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 変更する科学的変数は、HMM の遷移増分を exp226 geometry-only `tvt_geop` 差分から exp226 最終 `tvt_pred` 差分へ置き換えることだけとする。
- 最初の未知行では `tvt_pred[0] - last_known_TVT_input`、2 行目以降では `tvt_pred[t] - tvt_pred[t-1]` を使う。
- HMM の遷移中心には上記 TVT 増分をそのまま使う。rate state、rate smoothing、clipping、scale、segment 集約、momentum、残差 offset/rate state は追加しない。
- exp437 の TVT-only 状態空間、TVT grid、開始事前分布、位置遷移 kernel、typewell-GR emission、forward-backward、posterior mean readout は固定する。
- exp226 OOF は group-safe の保存済み最終予測を使い、未知 suffix の正解 TVT は候補予測と logical SHA を凍結するまで読まない。
- exp226 最終予測と HMM emission が同じ raw suffix GR を使うことは target leakage ではないが、証拠の二重利用による CV-to-hidden 汎化リスクとして明示する。
- Stage 0 は固定 32 well の機構確認であり、CV や昇格判定の代替にはしない。
- Stage 1 の full group-safe OOF は Stage 0 の全 gate 通過とユーザーの別承認後にだけ実行する。
- PF 版は exp491 の結果レビューとユーザー承認後、別 steering・別実験として設計する。exp491 内には PF 分岐を持ち込まない。
- 再現性は `docs/06_reproducibility.md` に従う。本実験候補に乱数はないが、入力 content SHA、stable row order、予測 content SHA、runtime 設定を記録する。
- 既存の正規 notebook scaffold は上書きせず、別名の Jupytext source /
  notebook に Stage 0 を実装する。実行 package は作らない。

## 計算量の事前登録

- 実装される Stage 0: candidate 1 variant × 32 well = 32 HMM well-run。
  保存済み exp226 final を比較対象とし、control HMM は再実行しない。
- Stage 1 承認後の想定: candidate 1 variant × 773 well = 773 HMM well-run。保存済み control は再実行しない。
- GPU 学習、LightGBM 学習、親実験の再学習はない。

## 受け入れ基準

- steering、実験ディレクトリ、`config.yaml`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json` が「Stage 0 実装済み・実行ロック中」で
  整合している。
- `KAGGLE_DIRECTION.md` に exp491 と条件付き PF 後続案が記録され、PF が exp491 の結果待ちでブロックされている。
- `experiment_summary.md` に exp491 が Stage 0 実装済み・未実行として記録されている。
- 変更変数、固定条件、入力 SHA、leakage guard、Stage 0/1 gate、停止条件、実行数が事前登録されている。
- compact self-contained Jupytext source / notebook と契約テストがある。
- 親 exp437 と同等以上の章構成を持ち、同一 exp helper import と
  `__file__` 依存がない。
- strict `usecols`、decompressed input SHA、truth/role/episode-late freeze、
  first-difference / TVT-U rate identity、posterior normalization、固定 gate が
  実装されている。
- Kaggle package、Kaggle run、submission は作成されていない。
- deterministic anchor とは主張しない。将来実行した場合は feature/input content SHA、prediction content SHA、Kaggle kernel version を記録する。
- gzip 生成物を比較する場合は raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録する。
