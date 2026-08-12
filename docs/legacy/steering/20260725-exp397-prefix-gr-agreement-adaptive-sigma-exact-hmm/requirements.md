# 要件

## 依頼

known prefix区間のhorizontal GRとtypewell GRの一致度をwellごとに計算し、その一致度に応じて
exp209 exact HMMのGR観測スケール（公開Notebookでの `gs`、本リポジトリでの
`sigma_gr`）を `1.0` または `1.3` 倍する実験を設計する。

このturnではbacklog、steering、実験ディレクトリと設計文書だけを作成する。
notebook、helper、test、Kaggle package、HMM実行、推論、提出は実装・実行しない。

## 制約

- 対象実験は `exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm`、Routeは `pf_beam`、
  親実験は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` に固定する。
- 変更する変数はwell単位の `sigma_gr` 係数だけとする。exp209のtypewell前処理、GR補間、
  base `sigma_gr`、Gaussian emission、state grid、transition、prior、posterior mean出力は固定する。
- 一致度はknown prefixのraw finite pairだけから計算するPearson相関 `rho_gr` とする。
  unknown suffix truth、error、Formation、hidden-like role、worst-well identityは係数決定に使わない。
- primary係数は `rho_gr >= 0.50` なら `1.0`、`rho_gr < 0.50` なら `1.3` とする。
  finite pairが64未満、どちらかの標準偏差が `1e-6` 以下、または相関がnonfiniteなら
  trusted parentへno-opとなる `1.0` をfallbackにする。
- base `sigma_gr` はexp209どおりknown prefixのzero-filled GR residual population stdを
  `[10, 60]` にclipする。係数はclip後に1回だけ掛け、再clipしない。
- Stage 0はtruth-free / 0-HMMの識別性・安定性監査とする。全gate PASS後もStage 1の実装・実行には
  別のユーザー承認を必要とする。
- Stage 1はscientific variant 1個、reporting fold 5、最大773 exact-HMM well-runs、
  model config・trained fold・LightGBM booster・PF・Beam・parent control再実行各0とする。
- 係数 `1.0` のwellは保存済みexp209 predictionをそのまま再利用し、`1.3` のwellだけを再計算する。
- multiplier、相関閾値、support、window、相関種、bias条件、clip、emission temperature、
  transition、prior、blend weightを結果後に変更して救済しない。
- exp307 / exp346のfinite-only scale縮小を再開せず、exp343のACF temperingやexp389のHuber救済にも
  位置付けない。独立したprefix agreement reliability仮説として扱う。
- inferenceとsubmissionはStage 1の全promotion gate PASS後も別承認まで無効とする。
- Kaggle Notebook実行を正とし、internet off / CPUで成立させる。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。

## 受け入れ基準

- `rho_gr` のpair、計算順、support/fallback、閾値、係数、base scaleとの合成順が一意に定義されている。
- Stage 0のcoverage、非退化、full/tail安定性gateと、Stage 1のglobal/fold/scope/by-well/fixed-blend
  promotion gateが実行前に固定されている。
- Stage 1の上限が1 variant / 5 reporting folds / 最大773 HMM well-runs / booster 0 /
  parent control再実行0で明記されている。
- 係数生成とcandidate predictionがunknown-suffix truth読込前にfreezeされる。
- exp209保存controlのdecompressed content SHA、row identity、prediction parityを検証する方針がある。
- stochastic componentなし、sorted well/row順、固定thread、logical/decompressed SHA方針が
  `docs/06_reproducibility.md` に沿っている。
- `KAGGLE_DIRECTION.md` と `experiment_summary.md` にdesign-only状態が記録されている。
- experiment scaffoldの設定・文書検証が通り、実装ファイルや実行成果物を追加していない。
