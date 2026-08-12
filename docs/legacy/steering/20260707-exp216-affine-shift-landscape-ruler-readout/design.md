# 設計

## アプローチ

既存の affine shift-scan 実装をベースに、`scan_filter_for_region` の出力を拡張する。
候補はそのままに、best/second/zero/secondary-mode の階層を明示し、`margin`、`zero_rank`、`bimodal_flag`、prefix holdout error を診断レポートで比較可能にする。

- compare surface: `raw`, `rolling_median_11`, `savgol_31_p2`
- compare calibration mode: `raw`, `flat_calibrated`, `heel_calibrated`
- metric: hidden-tail と prefix_backtest の row-level と aggregate
- readout: fixed exp072 candidate について観測コスト rank / gap を記録
- curve: 全行の raw curve は巨大になるため、surface / well / region / shift ごとの集約 curve を保存
- correlation: distance bucket ごとに ruler signal と絶対誤差の相関を保存

## 実験範囲

- 対象実験: `exp216_affine_shift_landscape_ruler_readout`
- Route: `pf_beam`
- 親実験: `affine_shift_landscape_ruler_readout` backlog
- 変更する変数: row-level readout columns, aggregate curve outputs, experiment settings
- 固定する変数: shift grid, scoring、candidate readoutフロー（replaceしない）

## 再現性設計

- seed policy: `no_rng_deterministic_linspace_sampling`
- stochastic 処理: 固定候補 cache を除きなし
- PF/Beam 再生成: なし
- seed bagging: なし
- CPU/GPU: CPU 実装、seed に依存しない決定的な処理
- train cache / feature regeneration SHA: cache input SHA と feature content SHA を保存
- model/prediction/submission SHA: 該当なし（train-side diagnostic）
- Kaggle bootstrap: `kaggle kernels push` 前後で kernel version と入力経路を確認

## リスク

- リークリスク: `zero_rank` が悪化しても top1 指標だけ改善に見える偽陽性。
- CV/LB 不一致リスク: これは LB 寄りではなく診断専用で、採否の補助指標としてのみ扱う。
- ランタイム/メモリリスク: `np.argsort` と shift curve 集約分は行数が多い井戸で増加するため、必要に応じて top-k 導出へ最適化。
- 再現性リスク: `exp072` cache の入力差分、Kaggle 実行環境のキャッシュ競合。
