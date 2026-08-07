# 設計

## 仮説

exp267 の離散 K=3 structure が再現しなくても、well 内 candidate divergence の大きさは
actual MAE 増加と calibration bias 低下を表す連続 risk として残る可能性がある。

## アプローチ

exp267 の `well_segment_signatures.parquet` と exp264 Stage B v2 の
`candidate_score_oof.parquet` だけを固定入力にする。score Parquet は 6 primitive candidates に限定して
streaming 集約し、well×candidate の actual MAE、predicted absolute error mean、calibration bias を作る。

主軸 `fixed_range_gap_axis` は 12 個の正方向 divergence 特徴を outer-train median、
RobustScaler `(25, 75)`、`[-10, 10]` clip 後に等重み平均する。secondary `pca1_axis` は全 18 特徴で
outer-train PCA1 を fit し、outer-train 上の主軸との相関が正になるよう符号を固定する。両軸とも
outer-valid の target、error、selector score を fit・符号・特徴選択に使わない。

各軸について candidate-bank 等重み平均と candidate 別の fold/pooled Spearman を報告する。
bootstrap は candidate-bank 平均だけを outer-fold 層化復元抽出し、主軸の actual MAE と
calibration bias に対する 95% percentile interval を主 guard に使う。PCA1 bootstrap は感度分析として
保存するが guard には使わない。

## 実験範囲

- 対象実験: `exp272_continuous_well_divergence_risk_readout_on_exp267`
- Route: `ensemble`
- 親実験: `exp267_well_segment_candidate_divergence_signature_cluster_on_exp265`
- score source: `exp264_exp263_candidate_confidence_dual_selector` Stage B v2 immutable dataset
- 変更する変数: K=3 離散 assignment を使わない連続軸 readout
- 固定する変数: 18 署名、fixed thirds、6 primitive candidates、outer 5 folds、score labels、
  RobustScaler quantile、clip、主軸 12 特徴、bootstrap 回数・区間・seed、guard

## 生成物

- `well_continuous_divergence_axes.parquet`
- `well_candidate_risk_metrics.csv`
- `well_divergence_readout_by_well.csv`
- `well_divergence_spearman.csv`
- `well_divergence_bootstrap_intervals.csv`
- `well_divergence_quantile_metrics.csv`
- `continuous_axis_preprocessors.json`
- `continuous_well_divergence_risk_readout.png`
- `readout_summary.json`
- `reproducibility_manifest.json`

## 再現性設計

- seed policy: bootstrap scope 名から SHA256 で stable seed を作り、`np.random.default_rng` を局所生成する。
- stochastic 処理の有無: well-bootstrap resampling のみ。PCA `svd_solver=full` は deterministic。
- PF/Beam / likelihood-PF / seed bagging: 再実行しない。保存済み score だけを読む。
- 並列処理と乱数の関係: bootstrap は single-thread loop、scope ごとに独立 stable seed。
- CPU/GPU runtime: Kaggle CPU、GPU/TPU/internet off。
- input SHA: exp267 signature byte/logical SHA、exp264 score byte SHA を照合する。
- feature/content SHA: OOF axes と by-well readout は logical frame SHA、preprocessor は canonical JSON SHA。
- model/prediction/submission SHA: model、prediction、submission を生成しないため対象外。
- Kaggle package bootstrap: canonical notebook と package support files の config/SHA 一致を push 前に確認する。
- deterministic anchor: false。train-side diagnostic であり、Kaggle rerun 前は固定 anchor と呼ばない。

## リスク

- リークリスク: PCA の符号や axis choice を outer-valid score で選ぶと漏洩する。主軸を事前固定し、
  PCA 符号は outer-train target-free 主軸だけで決める。
- 多重比較リスク: candidate 別・PCA1 結果は report-only とし、主軸 candidate-bank 平均だけを guard に使う。
- CV/LB 不一致リスク: これは新規予測を作らない診断で、CV/LB anchor を更新しない。
- ランタイム/メモリ: 411 MB score Parquet を batch streaming し、45M rows を一括ロードしない。
- 再現性リスク: bootstrap seed、input SHA、preprocessor JSON、logical content SHA を保存する。

## 次のアクション

実行承認後に canonical Kaggle CPU notebook を一度だけ実行し、primary guard を記録する。
PASS しても exp272 内では学習せず、別 add-only 実験の要否を判断する。
