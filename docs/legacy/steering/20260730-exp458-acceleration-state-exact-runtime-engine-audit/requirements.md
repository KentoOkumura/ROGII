# 要件

## 依頼

exp444の3状態acceleration exact HMMについて、科学仕様、全状態、遷移確率、
posterior readoutを変更せず、Kaggle private CPUの固定runtime上限内で実行できる
数式上同値な計算engineを別実験として設計する。

exp444はStage 0A runtime projection FAILのままterminal closeを維持する。
exp458はその結果を救済または再分類せず、「同じposteriorを十分高速に計算できる」
という独立したruntime実装仮説だけを検証する。

## 制約

- Route: `pf_beam`
- 科学仕様の構造参照: `exp444_acceleration_state_exact_hmm`
- root参照: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- scientific contract SHAはexp444 v1の
  `f4a0bbbcc8b9cb44a55cff29e07f49ed251e11a896b3e877b4e2d6f9d08f4972`
  と完全一致させる。
- acceleration 3状態、41 rate states、TVT格子、OU/position/emission/prior、
  forward-backwardのposterior mean/stdを変更しない。
- 許す変更は、float64 scaled probability-space演算、因子化/fusion、
  exact-bit `delta_MD` kernel cache、4-well外側並列だけとする。
- pruning、threshold、band/state削減、近似CDF、float32/float16、GPU、
  transition/emission/parameter変更を禁止する。
- Stage 0Aはexp444保存fixed4をload-only baselineとし、親HMMを再実行しない。
- Stage 0Aのcandidateは同じ4 wellsを2回実行する。scientific variant 1、
  runtime engine 1、candidate HMM well-runs 8、LightGBM config 0、
  trained fold 0、booster 0、parent/control rerun 0、PF 0、Beam 0、GPU 0。
- Stage 0B、Stage 1、inference、submissionはStage 0A PASS後も別承認対象とする。
- 2026-07-30のユーザー依頼によりcompact self-contained候補と専用testの実装、
  続く依頼により正規train Notebook採用、Kaggle package、Stage 0A実行を
  承認済みとする。
- 再現性: `docs/06_reproducibility.md` に従い、well/row/state順、parallel output順、
  CPU thread数、入力/source/prediction/posterior/runtime manifest SHAを固定する。

## 受け入れ基準

- fixed4のwell、row、state、scientific contract identityがexp444と完全一致する。
- exp444保存fixed4に対し、TVT posterior mean/stdの最大絶対差が`<=1e-5 ft`、
  acceleration posteriorが`<=1e-7`、rate diagnosticが`<=5e-6`である。
- small-trellis dense reference、OU/acceleration/position kernel、posterior
  normalization、finite coverageの事前固定gateをすべて満たす。
- 2回のcandidate outputはstable sort後にdecompressed content SHA、
  posterior bundle SHA、diagnostic SHAが完全一致する。
- 2回のうち遅いfixed4 decode wall timeを正とし、exp444 fixed4
  `746.353694 sec`比speedupが`>=4.75x`である。
- 遅いfixed4 decode wall timeから、fixed32を8 batch、full 773 wellsを
  194 batchとして投影し、それぞれ`<=3,600 sec`、`<=30,600 sec`である。
- peak RSSが`<=25 GB`、effective outer workersが4、worker内Numba/BLAS
  threadが各1である。
- truth、role、fold、episode、causeはprediction/posterior/diagnostic freeze前に
  一度も読まない。
- deterministic anchor として扱う場合は、input/source/scientific contract/
  prediction/posterior/diagnostic/runtime manifest SHAとKaggle kernel versionを
  記録する。modelとsubmissionは生成しないため非該当と明記する。
- gzip生成物はraw gzip SHAではなくdecompressed content SHAを主証拠にする。

## 次のアクション

正規train Notebookを採用し、Kaggle private CPUでStage 0Aを実行する。
Stage 0B/1、inference、submissionは各前提PASS後も別承認を得る。
