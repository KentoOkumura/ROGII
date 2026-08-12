# 設計

## アプローチ

`exp092` の OOF delta guard で作成済みの by-well / path-continuity readout を基準にし、通常 Kaggle notebook 実行で見える exposed sample / visible test の exp092 inference prediction に raw well context を付ける。

visible test には target がないため、guard は次を検査する。

- train/inference feature schema の完全一致。
- train/inference projection feature summary の source 別分布差。
- visible test well ごとの exp092 prediction step continuity。
- optional exp073 / exp077 inference surface がある場合の correction size と correction step。
- raw prefix anchor と prediction の last_known_tvt parity。
- OOF worst-regression wells の correction / step / range 分位点から見た visible-test outlier warning。

出力は `artifacts/worst_well_rawtest_guard/` に CSV / JSON として保存する。submission は作らない。この notebook は通常実行なので、Code Competition の hidden LB test は観測できない。

## 実験範囲

- 対象実験: `exp092_u_projection_correction_disagreement_fullrun`
- Route: `ml_model`
- 親実験: `exp092_u_projection_correction_disagreement_fullrun` follow-up
- 変更する変数: visible-test guard script / audit config / records
- 固定する変数: exp092 train artifacts、selected model `lgb1`、exp092 inference prediction、既存 submission

## 再現性設計

- seed policy: no new RNG。既存 exp092 inference artifact を読むだけ。
- stochastic 処理の有無: なし。学習、PF sampling、GPU 推論は実行しない。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。exp092 inference が保存した projection summary / prediction を監査する。
- 並列処理と乱数の関係: pandas groupby / merge の deterministic 集計のみ。
- CPU/GPU runtime と deterministic flags: CPU 集計のみ。LightGBM model はロードしない。
- train cache / visible-test feature regeneration の SHA 記録方針: schema / summary / prediction の raw SHA と gzip decompressed SHA を summary に保存する。
- model manifest / prediction / submission SHA 記録方針: model manifest は入力 metadata として SHA を記録する。submission は生成しない。prediction SHA は id order + float32 values で記録する。
- Kaggle package bootstrap 確認方針: notebook push は不要。script の py_compile / ruff / local smoke を通す。

## リスク

- リークリスク: visible test target は存在せず、submission も生成しないため低い。visible test distribution を見て提出後の説明に使うだけに留める。
- Code Competition 観測リスク: hidden LB test は code submission rerun 時にだけ差し替えられるため、この guard では観測できない。hidden 側で見たい事象は submission notebook 内 assert probe として別途設計し、pass/fail だけを信号として扱う。
- CV/LB 不一致リスク: target-free visible-test guard なので LB 改善や悪化を保証しない。警告は採用判断の補助として扱う。
- ランタイム/メモリリスク: 14k row 程度の inference prediction と小さな schema / summary を読むだけなので低い。OOF predictions 本体は読まず、既存 guard CSV を使う。
- 再現性リスク: 入力 Kaggle output の取得状況に依存する。見つけた入力 path と SHA を必ず記録する。
