# 要件

## 依頼

exp072 likelihood-PF の長い vertical offset が、HMM と同様の transition / prior
hysteresis なのか、PF 固有の particle extinction、resampling、seed aggregation、
support / clamp、GR emission のいずれで形成されるかを、考えられる限り多角的に
直接監査して原因を特定する。重い particle replay はローカルで実行せず Kaggle CPU
Notebook で実行する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親実験 exp072 の likelihood-PF を変更せず、500 particles、128 seeds、stable
  SHA256 seed、momentum / noise / clamp / ESS threshold / systematic resampling /
  roughening、seed arithmetic meanを固定する。
- 対象は固定済み exp072 `likpf_mean` で `abs(error) > 10 ft` が128行以上連続する
  PF-specific persistent-offset episodes とする。HMM episodeを流用しない。
- truth、error、episode境界は PF state transition、GR likelihood、weight normalization、
  resampling、prediction生成に入力しない。診断量の計算にだけ用いる。
- 診断 replay が固定予測を再現することを parity gate で確認する。
- 学習、LightGBM、fold学習、booster、GPU、inference、submissionは実行しない。
- 結果を見て排他的分類の優先順位や主要thresholdを変更しない。threshold sensitivityは
  独立表として併記する。
- ユーザーは2026-07-26に実行を許可し、重い実験はKaggleで実行するよう指定した。

## 受け入れ基準

- PF-specific episode / well / row数、asset SHA、固定 `likpf_mean` content SHAを記録する。
- 全対象wellで augmented replay と固定 `likpf_mean` の parityを確認し、最大絶対差、
  RMSE差、failed well数を記録する。
- predictive propagation、GR emission、resampling前後の truth-basin mass/count、
  ESS、resampling率、ancestor concentration、particle interval、seed trajectories、
  within-seed / across-seed平均の各診断を保存する。
- raw observed GR と interpolated/imputed GR を分離する。
- 初期条件、transition、emission、resampling extinction、within-seed averaging、
  across-seed aggregation、support/clamp、mixed/unresolved の排他的episode分類と、
  episode数・well数・row数・SSE比を出す。
- first escape / recapture、episode onset前128行、fold、tail距離、seed、threshold、
  error符号別の安定性を監査する。
- disable / counterfactual診断を、baseline再学習なしで可能な範囲に限定して実行し、
  「関連」だけでなく原因段階の切り分けを行う。
- full結果を読む前に固定したsentinel選択規則と12 paired variantsに従い、最大12 wells
  だけでinitialization、transition、GR sigma、resampling threshold、rougheningを
  個別介入する。GRはsigma 100万倍のnear-disable、clampはtypewell端の一定GR延長で
  marginを±200 ftへ広げる対照も含める。baseline parityとvariantごとの全suffix /
  固定episode SSEを保存する。
- arithmetic seed mean以外のtarget-free readoutは同一baseline particle bankから作り、
  suffix全体likelihoodを使うreadoutとtruth-best oracleをdeployable候補から分離する。
- counterfactual full前に1 sentinel well ×12 variantsのKaggle CPU preflightを完了し、
  augmented particle-mode baselineのpersisted parity 0と全variant artifactを確認する。
- Kaggle上の preflight と full run が成功し、必要なsummary / episode / cause /
  threshold / manifest生成物を取得・検証する。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md`、
  `KAGGLE_DIRECTION.md`を更新する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
