# exp308_imputed_gr_confidence_downweight

## 状態

- Route: `pf_beam`
- 状態: exp307 promotion gate FAILにより未実行のまま閉鎖
- 親: `exp307_finite_only_robust_sigma_gr`

## 仮説

線形補間値を実観測と同じ尤度で数える過信を、最近傍実測GRまでの距離に応じたsoft weightで緩和できる。

## 固定した変更

- observed: weight 1。
- imputed: `max(0.25, 2^(-distance/8))`。
- 補間GR値、finite-MAD sigma、typewell、HMM decoderはexp307固定。
- 1 variant、773 HMM runs、0 booster、parent再実行0。

## 検証方針

parentより0.05 ft以上・4/5 folds改善し、gap bucket、1000+、hidden-like、p95、worst、fixed blendを守る。exp269のblanket neutralityと異なりweight 0は禁止する。

## 実装

compact self-contained train/inference sourceと正規Notebookを実装した。trainはexp307 dependency preflight、raw mask/distance/weight freeze、weighted exact-HMM、truth late join、gap/distance readout、promotion gate、SHA保存をセルで追える。親gateがFAILしたためdependency値はpending sentinelのまま凍結せず、実行前に停止する。

## 結果

未実行。実装と12件のcontract testは履歴として保持するが、exp307 v2が全promotion gateをFAILしたため固定dependencyは成立しない。

## 所見

設計時点では精度所見はない。exp269の一律neutral化による悪化を踏まえ、実測GRのweight 1と補間GRのfloor 0.25を固定した。

## 次

実行、inference、submissionへ進まない。別parentへの再設計は独立した根拠と事前設計、ユーザー確認がある場合だけ扱う。
