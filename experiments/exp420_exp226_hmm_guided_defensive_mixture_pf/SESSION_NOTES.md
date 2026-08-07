# exp420_exp226_hmm_guided_defensive_mixture_pf セッションノート

## 目的

HMM、likelihood-PF、exp226の誤差原因から、absolute predictionをblendせず、
exp226 geometry rateとHMM rate innovation scheduleをimportance-corrected PF proposalへ
統合する。最終出力はPF 1 variantとし、routeを`pf_beam`へ固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: implementation complete / exp411 schedule prerequisite failed / not executed
- CV / LB / candidate prediction: まだなし
- 実装承認: あり（ユーザー指示 `exp420を実装してください`）
- 正規Notebook採用 / package / push / run / inference / submission承認: なし
- canonical train / inference Notebook: template placeholderのまま
- compact self-contained train候補 / 専用test: 実装済み
- Kaggle package: 未作成

## コマンドログ

2026-07-27:

- `make new-steering EXP=exp420_exp226_hmm_guided_defensive_mixture_pf`
- `make new-exp EXP=exp420_exp226_hmm_guided_defensive_mixture_pf SOURCE=templates/experiment`
- `kaggle-review-exp`の実験ライフサイクルと`docs/06_reproducibility.md`を確認
- `kaggle-strategy`でexp408 / exp410 / exp411 / exp419、現在のbacklogを確認
- steering、config、README、result、metrics、backlogをdesign-onlyとして作成
- fixed32はheaderを除き32行、PF sentinelは12行、well重複0をread-only確認
- `make update-summary`で`experiment_summary.md`へPF/Beam routeのexp420を追加
- `make validate-exp EXP=exp420_exp226_hmm_guided_defensive_mixture_pf`: strict PASS
- `review_exp_docs.py exp420 --root .`: collected files全体でcore evidence categoryあり
- steering / exp420配下（template `settings.py`を除く）の`TODO / TBD / FIXME`なし
- PF / HMM kernel、Jupytext source、test、compact候補Notebookを実装
- 正規Notebook、package、Kaggle runは0

### 実装

ユーザーの`exp420を実装してください`を、設計済み境界のうちtrain-side
Stage 0 / full実装承認として受け取った。正規Notebook採用、Kaggle package、
push、runは既存どおり別承認とした。

- `exp420_exp226_hmm_guided_defensive_mixture_pf_compact_selfcontained_train.py`
  をJupytext percent形式で実装し、compact候補Notebookへ変換した。
- exp209 / exp411互換のuntreated forward filterから、emission前後のrate mean差を
  fixed two-sided CUSUMへ渡し、PF実行前に`active / direction` scheduleをfreezeする。
  HMM backward pass、posterior mean、absolute stateは生成・利用しない。
- exp419 inactive proposalを維持し、active 32 transitionsだけgeometry 3成分各
  `1/12`と`mu0 + direction*0.005`中心HMM 3成分各`1/12`へ配分した。
  元transitionは常に`0.5`、importance correctionはclipなし`p0/q`、構成上上限2。
- all-guidance-zeroでexp404 RNG順、HMM-weight-zeroでexp419 RNG順を保つ分岐を実装し、
  synthetic bitwise parityを確認した。
- fixed32 / sentinel12はproposal前に`well`列だけをSHA固定で読み、重複なし44-well
  unionを作る。role / cause / episode / truth / foldはschedule、candidate、
  predictive-support SHAのfreeze後だけ読む。
- Stage 0はdirection、lead、control activation、sentinel episode SSE、
  majority-seed support外率、worst-wellをfail-close AND gateで判定する。
  selection-biased pooled RMSEはpromotionへ使わない。
- fullは4 deterministic LPT shards、strict merge、exp404 / exp226 / exp263 fixed
  physical blend、exp408 / exp410 episodes、hidden-like、scope / by-well tailを
  late joinし、mechanism / standalone / physical-anchor gateを順に判定する。
- model、booster、submissionは生成しない。inferenceはfail-closedのまま。

### 実装時の固定実行量

- active scientific variants: 1
- Stage 0 HMM signal / candidate PF well-runs: `44 / 44`
- Stage 0 seed-well trajectories / particle starts: `5,632 / 2,816,000`
- full HMM signal / candidate PF well-runs: `773 / 773`
- full seed-well trajectories / particle starts: `98,944 / 49,472,000`
- control HMM / PF / exp226 rerun: `0 / 0 / 0`
- reporting folds: 5
- LightGBM configs / trained folds / boosters / models / GPU:
  `0 / 0 / 0 / 0 / 0`

これは実装済みの固定実行量であり、Kaggle実行承認ではない。

### 検証

- exp420専用test: `13 passed`
- exp420 + exp419 + notebook / scaffold対象検証: `36 passed`
- all-guidance-zero synthetic exp404 RNG / prediction parity: bitwise一致
- HMM-weight-zero synthetic exp419 RNG / prediction parity: bitwise一致
- inactive / active importance ratio: finite、nonnegative、最大`<=2+1e-12`
- active schedule synthetic fixture: 7 proposal成分すべて使用
- no-trigger HMM fixtureでexp411 untreated forwardのpredictive / filtered rate、
  innovation、CUSUM、trigger / active、normalization、log-likelihoodが最大差0
- exp263 physical anchorは親contractどおり、exp072
  `last_known_tvt + likpf_mean_d`、exp209 `hmm_mean_tvt`、exp226 final OOFを
  `0.25 / 0.25 / 0.50`で再構成する。exp072差分列を絶対TVTとして扱わない。
- CUSUM trigger row自身はinactive、次のtransitionから固定3 rows activeとなる
  state-machine fixtureをPASS
- `jupytext --to ipynb --test`: PASS
- `py_compile`: PASS
- `ruff --select F821`: PASS
- `make validate-exp EXP=exp420_exp226_hmm_guided_defensive_mixture_pf`:
  strict PASS
- parent compact比較:
  exp419 `2,996`行に対しexp420は`5,203`行。13章でHMM schedule、scheduled
  proposal、fixed44/full、late truth、Stage 0/full gateをNotebook上で追える。
- `__file__`、同一exp helper import、exp419 / exact-HMM helper importは0。

## 変更点

- exp419 inactive proposalへ、固定HMM activation rowだけdirectional 3成分を追加する。
- active proposalは元transition `0.5`、geometry 3成分各`1/12`、
  HMM方向3成分各`1/12`。
- HMMはuntreated exp209 forward filterの`mu_filtered - mu_predictive`だけを使う。
- HMM CUSUMはexp411のdrift `0.01`、threshold `1.0`、activation `32`、
  refractory `128`を固定する。
- exp226はfold-safe `tvt_geop + Z`の局所rateだけを使う。
- HMM / exp226 absolute prediction、backward message、blend、selector、MLは使わない。
- PF target、x1.0 GR emission、500 particles、128 seeds、resampling、roughening、
  temperature-5 seed weightingを固定する。

## 設計根拠

- HMM: forward transition / prior hysteresisがepisode SSE `59.3978%`
- PF: finite support `36.4701%`、across-seed平均`36.2441%`
- exp226: donor局所rate mismatchを再anchorなしで累積

3者に共通するabsolute datumの弱さを避け、空間rate、target-well rate direction、
continuous PF supportという異なる強みだけをproposalへ使う。

## 固定実行量

Stage 0:

- scientific variants: 1
- fixed wells: 44（exp411 fixed32 + exp410 sentinel12、overlap 0）
- HMM signal / candidate PF well-runs: `44 / 44`
- control HMM / PF / exp226 reruns: `0 / 0 / 0`
- seeds / particles: `128 / 500`
- seed-well trajectories / particle starts: `5,632 / 2,816,000`

Full:

- scientific variants: 1
- HMM signal / candidate PF well-runs: `773 / 773`
- control HMM / PF / exp226 reruns: `0 / 0 / 0`
- seed-well trajectories / particle starts: `98,944 / 49,472,000`
- reporting folds: 5
- LightGBM configs / trained folds / boosters / models / GPU:
  `0 / 0 / 0 / 0 / 0`
- Kaggle CPU shards: 4

これは実装済みconfigの固定値であり、Kaggle実行承認ではない。

## 再現性メモ

- seed:
  `sha256("likpf::train::<well_id>") % 2147483647 + 1 + seed_index`
- stochastic components:
  proposal component draw、particle初期化 / propagation、systematic resampling、
  roughening
- HMM schedule:
  RNGなし。well / row順、CUSUM更新順を固定し、PF前にSHA freezeする
- parallel:
  well内single worker、shard順非依存。global RNGを使わない
- CPU/GPU:
  Kaggle private CPU、GPU off、internet off。fullは4 shards
- SHA:
  input、code、config、scientific contract、schedule、prediction、
  target-free diagnostic、well manifestを記録予定
- gzip:
  decompressed content SHAを主証拠とする
- model / submission SHA:
  modelとsubmissionを生成しないため非該当
- rerun:
  fixed probe wellでschedule / prediction / diagnostic logical parityを要求
- deterministic anchor:
  full coverage、全SHA、probe parity、raw-test regeneration前は主張しない

## 次のアクション

2026-07-28にexp411 Version 5の同一CUSUM schedule / fixed32 mechanism結果を回収した。

- future-rate direction agreement: `0.225397 < 0.60`
- passing folds: `0 / 5 < 4 / 5`
- control active-row fraction: `0.136119 > 0.10`
- persistent minus control active-well fraction: `0.0 < 0.20`

exp420は同じschedule、fixed32、direction / control gateをそのまま使うため、現行契約は
PF実行前にprerequisite FAILと判断する。正規Notebook採用、Kaggle package / push /
Stage 0 fixed44 / fullを行わない。compact実装は参照として保持する。再開にはscheduleを
そのまま使わない独立仮説、新しい実験設計、別承認が必要である。
