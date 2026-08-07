# exp389_exp209_huber_exact_hmm_emission セッションノート

## 目的

exp209 absolute-TVT exact HMMのGaussian emissionだけをfixed Huber
`delta=1.345`へ置換する本来の依頼を、exp357とは別の正しい親契約で設計確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU version 1完了 / tail gate FAIL / terminal close
- CV: `11.852741129500146`
- LB: まだなし

## 2026-07-25 完了確認と科学判定

- canonical kernel version 1（id_no `128466838`）は
  `KernelWorkerStatus.COMPLETE`。2026-07-24 23:38:19 UTCにCLIで再確認した。
- 科学summary生成時刻は2026-07-24 16:46:59 UTC、runtimeは
  `19,417.245940 sec`（約5時間23分37秒）。
- 実行量は設計どおり1 Huber variant / 773 HMM well-runs /
  reporting 5 folds / model・trained fold・booster・PF・Beam・control rerun各0。
- technical gateは全PASS:
  - 3,783,989 rows / 773 wells、finite coverage 1.0、ID mismatch 0。
  - posterior normalization最大誤差`3.55e-15`。
  - input preflight、Huber delta、exp209 sigma/clip、raw mask partitionをPASS。
  - truth access before prediction freezeは0 rows。
  - exp209 Gaussian controlは完全一致、LikPFおよびfixed 50:50 control parityは
    それぞれ`3.28e-6 / 3.64e-6 ft`で許容内。
- directはGaussian control `11.938287235`からHuber `11.852741130`へ
  `+0.085546105 ft`改善し、必要`+0.05 ft`を超えた。
- direct fold改善:
  - fold 0: `+0.004822 ft`
  - fold 1: `+0.050506 ft`
  - fold 2: `+0.117608 ft`
  - fold 3: `+0.240278 ft`
  - fold 4: `+0.006813 ft`
- required scope改善:
  raw observed `+0.111286`、raw missing `+0.030431`、
  high missing `+0.023166`、MD 1000+ `+0.083031`、
  hidden-like spatial `+0.079037`、typewell-purged `+0.014020 ft`。
- fixed LikPF/HMM 50:50は`10.269692505 -> 10.227661781`、
  `+0.042030724 ft`改善してPASS。
- by-wellは411/773改善、362/773悪化。delta中央値は
  `-0.000001 ft`だが、p95 `+0.002234 ft > 0`でFAIL。
  worst well `00bbac68`は`4.224995 -> 5.975244`、
  `+1.750248 ft > +0.25 ft`でFAIL。
- よってAND gateはFAIL、decisionは
  `huber_exp209_failed_close_without_rescue`。
  average/fold/scope改善があっても、少数wellのmode固定悪化を安全に抑えられない
  negative resultとしてterminal closeする。
- 再現性SHA:
  - scientific contract:
    `d685276820e999818aed316b6a67dc9c290f0c5b54b7f0bdbbc67fd9b430b165`
  - prediction raw gzip:
    `95302d547e8c49cdf67dabe6200e08e5c83f01ea158cf2fbd4f25b2fd1f74d75`
  - prediction decompressed/logical:
    `f5d44d9d9ee380bb7ea408006030363efbe8fcdb3573cfa18031b2d31c617f90`
  - raw-GR emission contract content:
    `660d72c9f67e04af6641ea7bde43057379169be9608a5149847e0f0c9befca63`
  - observation audit content:
    `d0a9fffcb8e16aacf4b03ee01bbc4fb1fd07bef4fb59df2135e2158ed5193c75`
  - input/control manifest:
    `89747fc5004a3b28d0d947981a0c54925518a16f221fc2bf8197bd73f15b7728`
  - promotion gate:
    `fe9dc5467120747847508eb60fe6e5bf45c3fb98b0070072d2c3e39a7e83271a`
  - overall/fold/scope metrics:
    `16831c2b5cf0c6dd74c7eb1619aaeb6b72445eeeb0db8138b86b346800c6c7f2`
  - by-well metrics:
    `b40199bd0b09a5c2a27a2b828f46b7fd7962a7ba2b79cbf152e39e4e6bab7bab`
- logsを一次証拠とし、不足したfold/scope/tailを確認するため評価JSON/CSV
  7ファイル、合計約0.38 MBだけを`/tmp`へ取得した。
  prediction 86 MBとraw-GR emission contract 20 MBは取得していない。
- delta/scale/clip/temperature/sigma/transition/grid/prior/blend救済、再実行、
  inference、submissionは行わず、同familyの新規backlogも追加しない。

## 2026-07-24 実行承認

- ユーザー指示「実行してください」により、正規train Notebook採用、
  Kaggle package/push/runを承認済みと記録した。
- 実行量をpush前に再確認した。
  - scientific variant: `1`
  - HMM well-runs / reporting folds: `773 / 5`
  - model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
  - PF / Beam / parent Gaussian control rerun: `0 / 0 / 0`
  - GPU/TPU/internet: off
- 保存済みexp209 Gaussian HMMとexp072 LikPFをload-only controlとして使い、
  親controlは再実行しない。
- canonical kernel:
  `kentookumura/exp389-exp209-huber-exact-hmm-emission-train`
- canonical title:
  `exp389 exp209 huber exact hmm emission train`
- kernel sources:
  - `kentookumura/exp209-joint-exact-parity-train`
  - `kentookumura/exp226-k16-kappa-repro-train`
  - `kentookumura/exp115-hidden-like-spatial-holdout-from-ppt-train`
- Kaggle CLI credentialはOAuth/legacy credentialを確認した。
- 3 kernel sourceの必要生成物をKaggle CLIで確認した。
  exp209 HMM/LikPF cache、exp226 reporting fold OOF、exp115 hidden-like
  assignmentはいずれも存在する。
- compact self-contained train候補を正規train Notebookへ採用した。
- raw-test inferenceとsubmissionは今回の承認に含めず、fail-closedを維持する。
- package metadataはcanonical id/title、private、CPU、internet off、
  run-on-push、competition source 1、kernel source 3を満たす。
- package notebookはbootstrap hash検証セルと最終
  `run_full_experiment(CONFIG)`セルを持つ。
- 正本 / loose package / bootstrap内config SHAは
  `f88b83bfd98a779b380cee9cf406ab8b176221f0812e147ae00b5e106ddb41ac`
  で一致した。
- 正規train Notebook SHA:
  `83f614c9d69c4cc8b8a1c918b5ac0f138ea3b33930aa0e41e2ba3f363a7aa0ba`
- push用train Notebook SHA:
  `bdf58b3c3881229de69b15454667acf1d3cedac9a58f4e22bbab2ee13164a548`
- kernel metadata SHA:
  `2cb22d5f82fe41de78e573e64fcc25b61b161e51dac2be4522d511cd1506f626`
- package後の専用9件 + 共通Kaggle Notebook 4件、strict experiment /
  project validation、RuffをPASSした。
- push前のcanonical kernel pullは`GetKernel 403 Forbidden`で、既存kernelを
  確認できなかった。新規canonical idへの初回pushとして扱い、別slugは作らない。
- 2026-07-24 11:22:04 UTCにcanonical kernel version 1をpushした。
- URL:
  `https://www.kaggle.com/code/kentookumura/exp389-exp209-huber-exact-hmm-emission-train`
- push後のpullでid_no `128466838`、private、CPU、internet off、
  competition source 1、kernel source 3、canonical id/titleを確認した。
- 起動直後および2026-07-24 11:25:34 UTCのstatusは
  `KernelWorkerStatus.RUNNING`。同時点のkernel logsは空で、実行中の同一versionを
  継続監視する。ログが空であることだけを理由に再pushしない。

## 2026-07-24 実装

- ユーザー指示「exp389を実装してください」をimplementation承認として記録した。
- exp374のcompact self-contained構成を参照し、科学的親はexp209のまま、
  行別emissionだけをfixed Huber `delta=1.345`へ置換した。
- 実装した内容:
  - exp209 known-prefix zero-fill population std、raw GR補間、Type Well GR、
    absolute-TVT grid、41 rate states、transition、prior、momentum、
    posterior meanの固定。
  - Huber
    `-0.5*z^2` (`|z|<=1.345`) /
    `-(1.345*|z|-0.5*1.345^2)` (`|z|>1.345`)。
  - input/control SHA、raw identity、fold/role identityのhard check。
  - prediction、raw-GR/emission contract、observation auditをcontent SHA付きで
    freezeした後だけtruth/controlをjoinする境界。
  - overall、5 folds、raw observed/missing、missing fraction、1000+、
    hidden-like 2面、by-well、fixed LikPF/HMM 50:50の固定readoutとAND gate。
- exp226 pre-freeze allowlistは`well_id,row_idx,suffix_offset,fold`だけで、
  `tvt_geop,tvt_pred,gr_delta,tvt_true,error,abs_error`はdecoderへ渡さない。
- exp357のshift-rank/exp281 residual-offsetコードを移植していない。
- helper importに依存しないcompact self-contained train候補と、明示停止する
  fail-closed inference候補をJupytext percent形式から生成した。
- 実装時点では既存の正規train/inference Notebookは明示承認なしに上書きせず、
  template placeholderのまま保持した。その後、実行承認により正規trainだけを
  採用し、正規inferenceはplaceholderのまま保持している。
- 専用テスト9件で次を確認した。
  - fixed Huber式と境界、追加clip/temperatureなし。
  - 同一emission入力に対するexp209 exact forward-backward kernel parity。
  - exp209 sigma/state grammar不変。
  - input SHA/identity allowlist、truth-late join。
  - 全事前gate、run未承認guard、fail-closed inference。
- 検証:
  - `py_compile`: PASS
  - Ruff `F821,F401,F841`: PASS
  - dedicated pytest: `9 passed`
  - dedicated + common Kaggle Notebook pytest: `13 passed`
  - Jupytext変換 / round-trip: PASS
  - strict experiment / strict project / template validation: PASS
  - 実験文書review: core evidence categories present
- repository全体は`905 collected / 895 passed / 6 skipped / 4 failed`。
  exp389専用9件は全PASSし、失敗4件は既存の実行状態期待差である。
  - exp296: status prefix 1件、`run_variant` / approval順序1件
  - exp384: `kaggle_execution_authorized`期待差1件
  - exp388: `approval_consumed`期待差1件
  本実験の変更対象外なので修正していない。
- 共通テスト初回指定で存在しない`tests/test_experiment_template.py`を指定し
  `no tests ran`となった。実在する`tests/test_kaggle_notebooks.py`へ訂正しPASSした。
- exp374 compact train 1,880行に対し、exp389 compact trainは同じ10章・
  1,884行で、route-specificな入力、decoder、評価、生成物を維持した。
- 実装完了時点ではKaggle package/push/run、ローカルNotebook実行、inference、
  submissionを実施していなかった。その後の実行承認によりKaggle private CPU
  train version 1だけを開始した。ローカルNotebook実行、inference、submissionは
  未実施である。

実装後も実行量契約は変更なし:

- scientific variant: `1`
- HMM well-runs / reporting folds: `773 / 5`
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / Beam / parent Gaussian rerun: `0 / 0 / 0`
- GPU/TPU/internet: off

## 2026-07-24 設計

- exp388まで使用済みを確認し、exp389を採番した。
- 親を`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`に固定した。
- exp357はexp281 residual-offset HMMを親にした誤スコープ履歴であり、
  本実験の親/control/入力/性能根拠から除外した。
- 同じexp209親のexp374を固定HMM契約の構成参照にしたが、Student-t式や結果は
  Huber delta/gate選択に使わない。
- Gaussian `-0.5*min(z^2,600)`からHuber `delta=1.345`へのrow emission
  単独置換を固定した。
- absolute TVT、grid/rate/transition/prior/sigma/missing/Type Well GR/
  momentum/likelihood weight/posterior meanはexp209を固定する。
- 0-HMM proxyは置かず、実行承認後に1 variant / 773 HMM runsを直接評価する。
- saved exp209 Gaussian `11.938287234887435`はSHA固定load-onlyで再実行しない。
- direct/fold/observed/missing/high-missing/1000+/hidden-like/by-well/fixed50:50の
  AND gateとno-rescueを固定した。

実行量契約:

- scientific variant: 1
- HMM well-runs / reporting folds: `773 / 5`
- model / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- PF / Beam / parent Gaussian rerun: `0 / 0 / 0`
- GPU/TPU/internet: off
- design時のimplementation / package / push / run / inference / submission承認: すべて0

## 変更点

- 本実験はexp209 Gaussian emissionだけをHuberへ変える。
- exp226はidentity/fold用途だけで、`tvt_geop`等をdecoderへ渡さない。
- 既存の正規notebookはtemplate scaffoldのままだが、別名のcompact
  self-contained候補に実験ロジックを実装済み。

## 再現性メモ

- `docs/06_reproducibility.md`確認済み。
- seed policy: RNGなし、sorted well/row/grid/rate/variant order。
- stochastic components: なし。
- CPU/GPU runtime: 将来のKaggle private CPUのみ、GPUなし。
- input SHA: exp209 HMM、exp072 LikPF、exp226 fold、exp115 hidden-likeの
  hard checkを実装済み。実値照合はKaggle run時に行う。
- prediction SHA: raw gzip / decompressed / logical contentの分離を実装済み。
  実値はKaggle run時に記録する。
- model / submission SHA: fitted model・submissionなしのため対象外。
- deterministic submission anchor: inference未実装なので主張しない。

設計検証:

- config YAML / metrics JSON parse: PASS
- strict experiment validation: PASS
- template / strict project validation: PASS
- experiment document review: core evidence categories present
- `experiment_summary.md`: exp389登録済み

## コマンドログ

```text
.venv/bin/python -m py_compile experiments/exp389_exp209_huber_exact_hmm_emission/*compact_selfcontained*.py
.venv/bin/ruff check experiments/exp389_exp209_huber_exact_hmm_emission/*compact_selfcontained*.py tests/test_exp389_exp209_huber_exact_hmm_emission.py --select F821,F401,F841
.venv/bin/pytest -q tests/test_exp389_exp209_huber_exact_hmm_emission.py
.venv/bin/pytest -q tests/test_exp389_exp209_huber_exact_hmm_emission.py tests/test_kaggle_notebooks.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp389_exp209_huber_exact_hmm_emission/*compact_selfcontained*.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp389_exp209_huber_exact_hmm_emission/*compact_selfcontained*.py
make validate-exp EXP=exp389_exp209_huber_exact_hmm_emission
.venv/bin/python scripts/validate_project.py --strict
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py exp389 --root .
make test
kaggle kernels status kentookumura/exp389-exp209-huber-exact-hmm-emission-train
kaggle kernels logs kentookumura/exp389-exp209-huber-exact-hmm-emission-train
kaggle kernels output kentookumura/exp389-exp209-huber-exact-hmm-emission-train --file-pattern '(overall_fold_scope_metrics|promotion_gate|scientific_contract|input_control_manifest|by_well_metrics|by_well_variant_runtime|observation_audit)'
make update-summary
```

## 次のアクション

本branchはterminal close済み。固定no-rescue契約に従い、再実行、inference、
submissionへ進まない。現行P1/P2の優先順位は変更しない。
