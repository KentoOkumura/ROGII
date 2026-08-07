# exp434_physics_candidate_public_lb_audit セッションノート

## 目的

`physical_model_summary.md`の12候補について、候補定義を変えずにPublic LBを
そろえ、OOF/LB順位の整合性を確認する。

## 現在の状態

- Route: `pf_beam`
- 状態: `completed_public_lb_census_12_of_12_scored`
- OOF: 親exp263の保存値を参照
- Public LB: 全12候補確定
- 実装: 承認済み・compact self-contained inferenceを正規Notebookへ採用済み
- 正規Notebook採用: 完了
- Kaggle package / push / run: version 1–10完了
- output取得 / submit-check: 10/10 PASS
- competition submission: version 1–10提出・採点完了、全10件COMPLETE

## 2026-07-29 設計セッション

### ユーザー依頼

ユーザーから、`physical_model_summary.md`でまとめた物理モデルOOFに加えてLBも
把握したいため、backlog、実験ディレクトリ、steeringを作り、実装前に設計を
確定するよう依頼された。

### 作成

- steering:
  `.steering/20260729-exp434-physics-candidate-public-lb-audit/`
- experiment:
  `experiments/exp434_physics_candidate_public_lb_audit/`
- route:
  `pf_beam`
- status:
  `design_frozen_not_implemented`

### 固定した設計

- 対象はexp263 Stage 1の6 primitive、5 pair、固定3-wayの12候補。
- OOFは3,783,989 rows / 773 wellsの保存値を使う。
- K16 9.837、LikPF 9.721、固定3-way 7.800は既存提出を候補同一性確認後に再利用。
- 未提出5 pairをbatch 1、未提出4 primitiveをbatch 2とする。
- 1日5件を超えず、通常は2日で`5 + 4`件を提出する。
- Batch 1の結果を見てもBatch 2の候補、順序、式を変更しない。
- K16 / LikPFの同一性gateが不合格なら既存LBを流用せず、同じ候補を追加提出する。
- model / config / trained fold / booster / GPU / parent再学習は0。
- LB結果によるweight grid、new blend、selector、candidate追加を禁止する。

### 実行量

- train variant / model config / trained fold / booster:
  `0 / 0 / 0 / 0`
- parent/control再学習:
  0
- planned inference candidate versions:
  9
- planned new submissions:
  通常9、最大11
- daily submission:
  通常`5 + 4`、最大`5 + 5 + 1`

### 現存する根拠

| 候補 | OOF | Public LB | ref | submission SHA |
| --- | ---: | ---: | ---: | --- |
| `exp226_k16` | 9.427110 | 9.837 | 54491603 | `b71e15f7...e906047` |
| `likpf_mean` | 11.594898 | 9.721 | 53706005 | `57d5c55c...8fa01` |
| `exp226_w500_50_50` | 8.238331 | 7.800 | 54761954 | `63166951...ce4b` |

K16とLikPFはexp434再生成値との同一性gate待ち。固定3-wayはexp263 v2/v3で
submission SHAがexactに一致している。

## 再現性メモ

- `docs/06_reproducibility.md`: 確認済み
- seed:
  42、PF familyはexp073 stable SHA256 per-well
- stochastic:
  PF/Beam/likelihood-PF raw hidden-test regeneration
- deterministic:
  K16、exact HMM、self-GR HMM、固定算術
- runtime:
  Kaggle CPU、GPU/internet off。親exposed runは354.341秒
- source SHA:
  exp073 / exp209 / exp223 / exp226をconfigへ固定
- prediction / submission SHA:
  candidate versionごとに記録予定
- model SHA:
  非該当
- bootstrap:
  metadata、embedded config、selected candidate、source inputをpush前に照合予定
- deterministic anchor:
  未実行のためfalse

## 2026-07-29 実装セッション

### ユーザー依頼と実装境界

- ユーザーの`exp434を実装してください`と対象名の確認を、設計済み
  `exp434_physics_candidate_public_lb_audit`の実装承認として扱った。
- 正規Notebook採用、Kaggle package / push / run、output取得、submit-check、
  competition submissionは別承認のまま変更していない。
- train variant / model config / trained fold / booster / parent再学習は
  `0 / 0 / 0 / 0 / 0`。Kaggle GPUも0。

### 実装

- Jupytext percent形式の
  `exp434_physics_candidate_public_lb_audit_compact_selfcontained_inference.py`
  と候補`.ipynb`を作成した。正規Notebookはplaceholderのまま。
- exp263 v3と同じexp073 PF/Beam/LikPF、exp209 exact HMM、exp223 self-GR
  HMM、exp226 K16 generatorをraw testへ適用する。
- generator 4本、exp226 source config、exp263 Stage 0 manifest、exp263
  Stage 1 formula bank、exp237 exposed reference gzipをSHA固定した。
- 6 primitiveのrow / well / ID / finite / duplicate / logical content SHA、
  5 pairと固定3-wayのfloat32 arithmetic parityを実装した。
- 12候補manifest、通常9候補だけのfail-close選択、fixed blend再提出拒否、
  K16 / LikPF同一性failure時だけの条件付き追加選択を実装した。
- 既存K16 / LikPF / fixed submissionはfile SHAを照合し、exposed ID集合が一致する
  runで数値同一性gateを実行する。hidden ID集合ではgateをskipとして記録する。
- `frozen_candidate_bank.parquet`、formula audit、candidate-version manifest、
  metrics、`submission.csv`へsource / prediction / submission SHAとfallback
  rows 0を保存する。
- package / run時は正規Notebook採用、package、push、run、selected candidate、
  candidate version label、package notebook SHAがそろわなければ停止する。

### Notebook構成比較

- 親exp263 inference sourceは647行・8章。exp434 compact候補は1,493行・
  11章で、source/config/SHA preflight、6 primitive生成、12候補式、
  existing equivalence gate、1候補選択、manifest保存までをセル上で追える。
- `__file__`、同一exp helper import、`from module import main; main()`は0。

### 検証

- exp434専用contract test: `8 passed`
- `py_compile`: PASS
- `ruff --select F821`: PASS
- `jupytext --to ipynb --test`: PASS
- exp263 + exp434関連test: `22 passed`
- 保存済みexp263 v3 exposed formula bankを6 primitive入力として再構成し、
  5 pair + fixedを含む全12候補で最大差`0.0 ft`を確認した。
- 保存済み既存submissionとの実装gateをread-only確認し、fixed 3-wayは
  最大差`0.000484375 ft`、K16は`0.000488265 ft`で、ともに`0.001 ft`
  gateをPASSした。LikPFはlocal output未保持のためKaggle exposed runで確認する。
- competition dataを使うローカルNotebook実行: 0
- Kaggle package / push / run / submission: 0

### Repo全test

`make test`は1,419件のcollection中、exp297 / exp301 / exp333 / exp336 /
exp349の既存5 test moduleが各自のconfig locatorで別実験configを読み、
scientific contract mismatchとなってcollection errorで停止した。exp434 testは
単独8件、親exp263との関連22件をすべてPASSしており、この5件はexp434の
tracebackや変更ファイルを含まないため、本実装のfailureとは扱わない。

## 未実施

- competition submission / scoring監視
- 12候補LB集計
- `physical_model_summary.md`へのLB列反映

## 次のアクション

competition submissionの別承認後、submit-check済み10候補を事前登録済み順序と
1日上限に従って提出する。scoring完了後にPublic LB、ref、elapsedを記録し、
12候補のOOF/LB順位診断と`physical_model_summary.md`更新を行う。

## 2026-07-29 Kaggle candidate generation

### 実行結果

- 正規Notebook:
  `exp434_physics_candidate_public_lb_audit_inference.ipynb`
- canonical SHA:
  `00c31d80a3b817445fe6d1d3cc6b199b776ad05bd0d43e694b0f946706a9a95f`
- Kaggle kernel:
  `kentookumura/exp434-physics-candidate-lb-audit-infer`
- 通常version:
  1–9、5 pair + 4 primitive
- 条件付きversion:
  10、`likpf_mean`
- Kaggle status:
  10/10 `COMPLETE`
- output取得 / submit-check:
  10/10 PASS
- candidate bank SHA:
  `870f0795649ee21852679176f313efb668bf7aec0c3262681a17c04b33eca03d`
- 共通監査:
  14,151 rows / 3 wells / fallback 0、全12式の親parity最大差0.0 ft、
  exposed reference gate PASS
- 学習量:
  0 model config / 0 fold / 0 booster / 0 parent retraining / 0 GPU

### 既存LB同一性gate

- `exp226_k16`:
  最大差`0.000488265 ft`、`0.001 ft`以内でPASS。既存Public LB
  `9.837`を再利用可能。
- `exp226_w500_50_50`:
  最大差`0.000484375 ft`、PASS。既存Public LB`7.800`を再利用可能。
- `likpf_mean`:
  exp069 v3提出との差が最大`4.7783203125 ft`でFAIL。既存Public LB
  `9.721`はexp434候補へ流用しない。事前登録済みfailure policyに従って
  version 10で同じ`likpf_mean`を生成し、submit-checkまでPASSした。

### version台帳

詳細なpackage / prediction / submission SHAは`kaggle_run_ledger.json`を正とする。

| v | candidate | OOF RMSE | submission SHA | 状態 |
| ---: | --- | ---: | --- | --- |
| 1 | `exp226_k16__selfgr_hmm_a070` | 8.532715 | `1620deb2...e12` | COMPLETE / check PASS |
| 2 | `exp226_k16__exact_hmm` | 8.635074 | `b3018e7c...3449` | COMPLETE / check PASS |
| 3 | `exp226_k16__likpf_mean` | 8.813822 | `60ba1203...e1cb0` | COMPLETE / check PASS |
| 4 | `selfgr_hmm_a070__likpf_mean` | 10.123457 | `d38f626a...8545e` | COMPLETE / check PASS |
| 5 | `likpf_mean__exact_hmm` | 10.269697 | `804a497e...60f6` | COMPLETE / check PASS |
| 6 | `selfgr_hmm_a070` | 11.349943 | `3ab78ac6...6c2d2` | COMPLETE / check PASS |
| 7 | `exact_hmm` | 11.938287 | `b446f2fd...9b418` | COMPLETE / check PASS |
| 8 | `pf_ancc` | 14.493051 | `8e188be7...1c2a0` | COMPLETE / check PASS |
| 9 | `beam_mean` | 15.774327 | `1c019099...e92e` | COMPLETE / check PASS |
| 10 | `likpf_mean` | 11.594898 | `026f16a3...ff99` | conditional / COMPLETE / check PASS |

competition submissionは承認されておらず、0件のままである。

## 2026-07-29 LikPF existing-equivalence failure root cause

### 結論

exp069 v3とexp434の`likpf_mean`は表示名とPF本体の式・粒子数・seed数は同じだが、
per-well `seed_base`の生成契約が異なる。したがって両者は同じ確率的候補の
再実行ではなく、別のMonte Carlo realizationである。exp069 Public LB `9.721`を
exp263/exp434候補のLBとして流用できないというequivalence gate判定は正しい。

### コード差

- exp069 v3 source SHA:
  `3dfa8c179ce195c092cd67ae065e8d5af1c4ddb4e7de62756c2c457416b5246a`
- exp073/exp434 source SHA:
  `4af212a8a1c83e36cdcc0bc912942a62df1fbc94ca67fd75789171afaa1a647e`
- `_pf_lik_allseeds`と`lik_pf`の関数本体は両sourceで同一。
- exp069 v3:
  BLAKE2b-8、key=`test:{well}:likpf`、little-endian、
  modulo=`2_147_000_000`、deterministic時`n_jobs=1`。
- exp073/exp263/exp434:
  SHA256、key=`likpf::test::{well}`、hex先頭16桁、
  modulo=`2_147_483_647`後に`+1`、`n_jobs=8`。
- `_pf_lik_allseeds`は各seedで`np.random.seed(seed_base + s)`を実行するため、
  `seed_base`が変わると128本すべてのparticle pathが変わる。

current-test 3 wellsのseedは次のとおりで、全件不一致だった。

| well | exp069 BLAKE2b seed | exp073/exp434 SHA256 seed |
| --- | ---: | ---: |
| `000d7d20` | 1,377,345,681 | 805,188,988 |
| `00bbac68` | 222,083,369 | 829,597,097 |
| `00e12e8b` | 254,364,174 | 1,365,511,604 |

### 提出差分

exp069 v3 submission SHA
`57d5c55c5caa1d07b6691a054116b434d63dd9f8e03c73dfb6ef45753aa8fa01`
とexp434 v10 candidate bankをIDでone-to-one整列した。

- rows:
  14,151、ID集合一致
- changed rows:
  14,148
- exp434 - exp069 mean:
  `-1.044576 ft`
- diff RMSE:
  `1.723127 ft`
- max abs:
  `4.7783203125 ft`
- correlation:
  `0.999989`
- well別diff RMSE:
  `000d7d20=0.046256`、`00bbac68=2.360595`、
  `00e12e8b=1.405478 ft`
- suffix quartile別diff RMSE:
  `0.605804 / 0.662212 / 1.600274 / 2.917223 ft`

suffix後半ほど差が拡大しており、seedの違うparticle pathの累積的な乖離と整合する。
ID順、schema、fallback、pair算術が原因ではない。exp434 submissionと同じversionの
frozen bankとの差はCSV round-trip由来の最大`0.000484 ft`だけで、gate failureより
4桁小さい。

### `n_jobs`について

exp069は1、exp073/exp434は8でありruntime契約も異なる。ただしexp434の10 versionは
すべて同一candidate bank SHA
`870f0795649ee21852679176f313efb668bf7aec0c3262681a17c04b33eca03d`
を生成し、exp073も同一package rerunのfeature content SHA一致を記録している。
今回の不一致を説明する十分かつ直接の差はseed生成規約である。same-seedで
`n_jobs=1`対8だけを変えるmatched runは実施していないため、並列数単独の
bitwise parityは本調査では主張しない。

## 2026-07-29 competition submission batch 1（3枠）

ユーザーから本日の残り3枠について明示承認を得た。凍結済み順序の先頭3候補を
再生成せず、取得済みKaggle outputとkernel versionを使って提出した。

- 提出直前check:
  3件とも14,151 rows / `id,tvt` / ID重複なし / NaN・Infなし /
  sample header・row count一致。submission SHAもversion台帳と一致。
- kernel:
  `kentookumura/exp434-physics-candidate-lb-audit-infer`
- competition:
  `rogii-wellbore-geology-prediction`
- 提出:

| 順序 | candidate | kernel v | ref | status |
| ---: | --- | ---: | ---: | --- |
| 1 | `exp226_k16__selfgr_hmm_a070` | 1 | `55083262` | PENDING |
| 2 | `exp226_k16__exact_hmm` | 2 | `55083266` | PENDING |
| 3 | `exp226_k16__likpf_mean` | 3 | `55083270` | PENDING |

最新v3は`kaggle-submit-monitor`でPENDINGを確認し、
`logs/submission_exp434_batch1_v3.log`へ記録した。先行提出もPENDINGのため、
Public LB確定後に3件まとめてmetrics/result/submission履歴を更新する。
残り未提出は凍結順序4–10の7候補である。

## 2026-07-30 competition submission batch 1 採点完了

Kaggle CLIでref、description、status、Public LBを照合し、3件すべて
`COMPLETE`を確認した。最新v3は`kaggle-submit-monitor`でも完了を記録した。

| candidate | OOF RMSE | Public LB | LB - OOF | ref |
| --- | ---: | ---: | ---: | ---: |
| `exp226_k16__selfgr_hmm_a070` | 8.532715 | 7.913 | -0.619715 | `55083262` |
| `exp226_k16__exact_hmm` | 8.635074 | 7.678 | -0.957074 | `55083266` |
| `exp226_k16__likpf_mean` | 8.813822 | 8.365 | -0.448822 | `55083270` |

- batch 1 LB首位:
  `exp226_k16__exact_hmm`、`7.678`
- batch 1内のOOF首位:
  `exp226_k16__selfgr_hmm_a070`、`8.532715`
- 既存LB確定2候補を含む暫定5候補のSpearman順位相関:
  `0.700`
- 解釈:
  exact HMMとのpairはPublic splitで相対的に強く、self-GR HMM pairとの
  OOF順位が逆転した。ただし残り7候補のLB観測前なので最終結論やweight変更は
  行わない。
- 次:
  凍結順序4–10の7候補を別日の提出枠で提出する。

## 2026-07-30 competition submission batch 2（4枠）

ユーザーから「次の4つだけ」の明示承認を得た。凍結順序どおりversion 4–7を
再生成せず、取得済みKaggle outputから提出した。version 8以降は提出していない。

- 提出直前check:
  4件とも14,151 rows / `id,tvt` / ID重複なし / NaN・Infなし /
  sample header・row count一致。submission SHAもversion台帳と一致。
- kernel:
  `kentookumura/exp434-physics-candidate-lb-audit-infer`
- competition:
  `rogii-wellbore-geology-prediction`
- 提出:

| 順序 | candidate | kernel v | ref | status |
| ---: | --- | ---: | ---: | --- |
| 4 | `selfgr_hmm_a070__likpf_mean` | 4 | `55105249` | PENDING |
| 5 | `likpf_mean__exact_hmm` | 5 | `55105256` | PENDING |
| 6 | `selfgr_hmm_a070` | 6 | `55105261` | PENDING |
| 7 | `exact_hmm` | 7 | `55105266` | PENDING |

最新v7は`kaggle-submit-monitor`でPENDINGを確認し、
`logs/submission_exp434_batch2_v7.log`へ記録した。採点完了後に4件のPublic LBを
metrics/result/physical model summary/提出履歴へ反映する。残り未提出は
version 8 `pf_ancc`、version 9 `beam_mean`、version 10 `likpf_mean`の3候補である。

## 2026-07-30 competition submission batch 2 採点完了

Kaggle CLIでref、description、status、Public LBを照合し、4件すべて
`COMPLETE`を確認した。最新v7は`kaggle-submit-monitor`でも完了を
`logs/submission_exp434_batch2_v7.log`へ追記した。

| candidate | OOF RMSE | Public LB | LB - OOF | ref |
| --- | ---: | ---: | ---: | ---: |
| `selfgr_hmm_a070__likpf_mean` | 10.123457 | 8.812 | -1.311457 | `55105249` |
| `likpf_mean__exact_hmm` | 10.269697 | 8.642 | -1.627697 | `55105256` |
| `selfgr_hmm_a070` | 11.349943 | 9.318 | -2.031943 | `55105261` |
| `exact_hmm` | 11.938287 | 9.063 | -2.875287 | `55105266` |

- 既存2候補と新規7候補を合わせた採点済み候補:
  9 / 12
- 採点済み9候補のSpearman順位相関:
  `0.750`
- primitive順位:
  OOFでは`exp226_k16 < selfgr_hmm_a070 < exact_hmm`だったが、LBでは
  `exact_hmm 9.063 < selfgr_hmm_a070 9.318 < exp226_k16 9.837`へ逆転した。
- LikPF pair順位:
  OOFではself-GR pairが上だったが、LBではexact-HMM pair
  `8.642`がself-GR pair `8.812`を上回った。
- 現在のLB首位:
  `exp226_k16__exact_hmm`、`7.678`
- 残り未提出:
  version 8 `pf_ancc`、version 9 `beam_mean`、version 10 `likpf_mean`

## 2026-07-31 competition submission batch 3（3枠）

ユーザーから「次の3つ」の明示承認を得た。凍結順序どおりversion 8–10を
再生成せず、取得済みKaggle outputから提出した。これで生成済み10候補は
すべてcompetition submission済みとなった。

- 提出直前check:
  3件とも14,151 rows / `id,tvt` / ID重複なし / NaN・Infなし /
  sample header・row count・ID順序一致。submission SHAもversion台帳と一致。
- kernel:
  `kentookumura/exp434-physics-candidate-lb-audit-infer`
- competition:
  `rogii-wellbore-geology-prediction`
- 提出:

| 順序 | candidate | kernel v | ref | status |
| ---: | --- | ---: | ---: | --- |
| 8 | `pf_ancc` | 8 | `55133068` | PENDING |
| 9 | `beam_mean` | 9 | `55133072` | PENDING |
| 10 | `likpf_mean` | 10 | `55133074` | PENDING |

最新v10は`kaggle-submit-monitor --once`でPENDINGを確認し、
`logs/submission_exp434_batch3_v10.log`へ記録した。未提出候補は0件である。
3件の採点完了後にPublic LB、最終12候補順位、gap、Spearman順位相関を更新する。

## 2026-07-31 competition submission batch 3 採点完了

Kaggle CLIでref、description、status、Public LBを直接照合し、3件すべて
`COMPLETE`を確認した。確認時点では、より新しいexp494 ref `55134873`が最新行で
scoring中だったため、latest-rowだけを読むmonitor scriptは誤帰属回避のため使わず、
exp434のref 3件を個別に照合した。

| candidate | OOF RMSE | Public LB | LB - OOF | ref |
| --- | ---: | ---: | ---: | ---: |
| `pf_ancc` | 14.493051 | 12.061 | -2.432051 | `55133068` |
| `beam_mean` | 15.774327 | 15.563 | -0.211327 | `55133072` |
| `likpf_mean` | 11.594898 | 9.807 | -1.787898 | `55133074` |

- 全候補の採点:
  12 / 12。exp434からのcompetition submissionは10 / 10 COMPLETE。
- 全12候補のOOF/LB Spearman順位相関:
  `0.846154`。
- LB首位:
  `exp226_k16__exact_hmm`、`7.678`。固定3-way `7.800`、K16 + self-GR
  HMM `7.913`が続いた。
- primitive順位:
  `exact_hmm 9.063 < selfgr_hmm_a070 9.318 < likpf_mean 9.807 <
  exp226_k16 9.837 < pf_ancc 12.061 < beam_mean 15.563`。
- 最大の順位逆転:
  K16がOOF 5位からLB 10位、exact HMMがOOF 10位からLB 7位。
- LikPF:
  今回のSHA256 seed版は`9.807`。exp069 BLAKE2b seed版`9.721`との差は
  `+0.086`だが、別Monte Carlo realizationなので同一候補として流用しない。
- 判断:
  Public LBは記述的なcensusとして完了し、weight tuning、candidate追加、
  train-side採用、最終提出への自動昇格には使わない。
