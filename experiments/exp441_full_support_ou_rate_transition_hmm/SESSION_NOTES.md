# exp441_full_support_ou_rate_transition_hmm セッションノート

## 目的

exp209のrate状態を残しながら、隣接3状態kernelの人工的な追従速度制限だけを除く。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle private CPU Stage 0 version 1完走、`stage0_fail_closed`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 実装承認: 2026-07-29のユーザー依頼「exp441を実装してください」
- 正規train Notebook採用 / Kaggle package / Stage 0 push/run:
  2026-07-30のユーザー依頼「実行してください」で承認済み
- Stage 1 / rerun / inference / submission: gate FAILにより閉鎖
- CV / LB: なし

## 2026-07-30 Stage 0実行承認

- scientific candidate: 1
- candidate exact-HMM well-runs: 32
- reporting folds: 5
- saved exp209 control rerun: 0
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`
- target: Kaggle private CPU、internet disabled
- 正規train Notebook採用、package、push/runのみ承認済み。
- Stage 1、inference、submissionは対象外。

正規train Notebook採用:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  --output experiments/exp441_full_support_ou_rate_transition_hmm/\
exp441_full_support_ou_rate_transition_hmm_train.ipynb \
  experiments/exp441_full_support_ou_rate_transition_hmm/\
exp441_full_support_ou_rate_transition_hmm_compact_selfcontained_train.py
```

- compact / canonicalとも24 cells、2,979 source lines。
- cell source SHA:
  `9f79f98ae65491669ff6547d81e28772af93520e953ae24e8d01603541d8c994`
- inference Notebookは未変更。
- exp209 / exp408のKaggle kernel source metadata pullはPASS。
- exp441 canonical IDのpush前pullは403で、既存kernelを確認できなかった。
  canonical IDは変更せず、同じIDへ初回pushする。

Kaggle package:

```bash
make prepare-kaggle-notebooks \
  EXP=exp441_full_support_ou_rate_transition_hmm \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp441-full-support-ou-rate-transition-hmm-train \
  --title 'exp441 full support ou rate transition hmm train' \
  --run-on-push --strict --no-src"
```

- private: true
- CPU / TPU / internet: false / false / false
- competition source: `rogii-wellbore-geology-prediction`
- kernel sources:
  `kentookumura/exp209-joint-exact-parity-train`、
  `kentookumura/exp408-hmm-message-rate-basin-audit-train`
- self-contained実装のため未使用repository `src/`は埋め込まない。
- bootstrap: 8 files。固定3 assetの展開後SHA、Stage 0承認、
  Stage 1/inference無効、CPU設定を検証してPASS。
- package notebook SHA:
  `41251a37286a8aad772503239112e0dc3cb07c897fc64055e27bb583856ec5d2`
- kernel metadata SHA:
  `0cbff8645f9932c1ed881d5c71fe26315a20f93f91d1d415e19eb7cc9dfd3656`
- 実行承認更新後の専用pytest `15 passed`、exp408/411/441関連pytest
  `41 passed`。Jupytext、py_compile、Ruff、strict experiment/template
  validationもPASS。

### Kaggle Stage 0 version 1

- kernel:
  `kentookumura/exp441-full-support-ou-rate-transition-hmm-train`
- kernel id_no: `129095333`
- URL:
  `https://www.kaggle.com/code/kentookumura/exp441-full-support-ou-rate-transition-hmm-train`
- version: 1
- started確認: `2026-07-29 21:07:59 UTC`
- completion status確認: `2026-07-29 22:14:23 UTC` /
  `KernelWorkerStatus.COMPLETE`
- push後のmetadata pullでcanonical ID、private、CPU、internet無効、
  exp209/exp408 kernel sourceを再確認した。
- scientific candidate 1、32 HMM well-runs、control再実行0、
  model/booster/PF/Beam/GPU各0。

#### Stage 0結果

- decision: `stage0_fail_closed`
- rows / wells / reporting folds: `156,088 / 32 / 5`
- notebook elapsed / peak RSS:
  `1,582.080005 sec / 1.123249 GB`
- technical gate: `16 / 17 PASS`
  - FAIL: full 773-well runtime projection
    `38,217.120130 sec > 30,600 sec`
  - analytic OU mass/moment、dense brute-force、position parity、
    normalization、finite coverage、SHA readback、truth-late、RSSはPASS。
- mechanism gate: `2 / 7 PASS`
  - PASS: matched-control pooled RMSE delta
    `-0.061891 ft <= +0.02 ft`
  - PASS: matched-control by-well delta p95
    `+0.037121 ft <= +0.25 ft`
  - FAIL: zero-directed under-response SSE share削減
    `0.022974 < 0.05`
  - FAIL: forward-cause episode SSE削減
    `-0.001635 < 0.10`
  - FAIL: persistent episode SSE削減
    `-0.016743 < 0.05`
  - FAIL: persistent改善well
    `8 / 16 < 10 / 16`
  - FAIL: persistent改善fold
    `1 / 5 < 4 / 5`
- Stage 1 eligible: false
- fixed32はmechanism-onlyであり、CV / promotion evidenceではない。
- fail actionどおりOU parameter、`sig_r`、momentum、support、emission、
  grid、gateを救済しない。

再現性:

- scientific contract SHA:
  `c38318d03a5290e5d5f2fc82e05c8649216c8bf1d6948bbf9633aa49cc9105c4`
- combined transition kernel manifest SHA:
  `6448b4e8a74f0bd4f670e3c8a1fe872b42f88d1bfe635f8bc57ade66765efc4b`
- prediction manifest / decompressed content SHA:
  `d7bbd2fe08957564575da25f2aa2297170cd3c6be39bfec15608ce122ab96511` /
  `063ff78b6a5e352681391bf37c1eecec2f841fe477629afa70b36b2065f13c92`
- diagnostic manifest / decompressed content SHA:
  `331ee69054769d631d5aeebfd918135eee7d08c08ef9c576704db99490665fcf` /
  `7bea35d0e190c68db8df619ccd295c0287d6bd294fa5abf2ee4bdd955a2b0a61`
- actual runtime versions:
  Python 3.12.13、NumPy 2.0.2、pandas 2.3.3、Numba 0.60.0。
- Kaggle outputは結果記録に必要な`metrics.json`だけを取得した。

## コマンドログ

### 2026-07-29 design-only作成

```bash
make new-steering EXP=exp441_full_support_ou_rate_transition_hmm
make new-exp EXP=exp441_full_support_ou_rate_transition_hmm
```

### 2026-07-29 compact self-contained実装

作成:

- `exp441_full_support_ou_rate_transition_hmm_compact_selfcontained_train.py`
- `exp441_full_support_ou_rate_transition_hmm_compact_selfcontained_train.ipynb`
- `exp441_full_support_ou_rate_transition_hmm_compact_selfcontained_inference.py`
- `exp441_full_support_ou_rate_transition_hmm_compact_selfcontained_inference.ipynb`
- `experiments/exp441_full_support_ou_rate_transition_hmm/tests/test_exp441_full_support_ou_rate_transition_hmm.py`

検証:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp441_full_support_ou_rate_transition_hmm/*compact_selfcontained*.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp441_full_support_ou_rate_transition_hmm/\
exp441_full_support_ou_rate_transition_hmm_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp441_full_support_ou_rate_transition_hmm/\
exp441_full_support_ou_rate_transition_hmm_compact_selfcontained_inference.py
.venv/bin/python -m py_compile \
  experiments/exp441_full_support_ou_rate_transition_hmm/*compact_selfcontained*.py
.venv/bin/ruff check \
  experiments/exp441_full_support_ou_rate_transition_hmm/*compact_selfcontained*.py \
  experiments/exp441_full_support_ou_rate_transition_hmm/tests/test_exp441_full_support_ou_rate_transition_hmm.py
.venv/bin/pytest -q experiments/exp441_full_support_ou_rate_transition_hmm/tests/test_exp441_full_support_ou_rate_transition_hmm.py
.venv/bin/pytest -q \
  experiments/exp408_hmm_message_rate_basin_audit/tests/test_exp408_hmm_message_rate_basin_audit.py \
  experiments/exp411_predictive_filtered_rate_innovation_destick/tests/test_exp411_predictive_filtered_rate_innovation_destick.py \
  experiments/exp441_full_support_ou_rate_transition_hmm/tests/test_exp441_full_support_ou_rate_transition_hmm.py
make validate-exp EXP=exp441_full_support_ou_rate_transition_hmm
make validate-template
```

結果:

- 専用pytest: `15 passed`
- exp408/411/441関連pytest: `41 passed`
- Jupytext `--test`: train / inferenceともPASS
- `py_compile`: PASS
- Ruff全選択ルール: PASS
- strict experiment validation: PASS
- template validation: PASS
- ローカル環境にはNumbaがないため、専用pytestではNumba decoratorを
  identity stubとしてcontractを検証した。Kaggle Stage 0ではNumba 0.60.0で
  JIT compile/runtimeを完走した。
- 親compact比較:
  - exp439: 9章、3,125行
  - exp440: 10章、2,576行
  - exp441: 10章、3,040行
- exp441はinput、OU kernel、exact HMM、brute-force、truth-late、
  gate、metrics/orchestrationをNotebook上で追え、親compactより薄い
  entrypointではない。
- `__file__`: compact train/inferenceとも0件。
- 実装時点では正規`*_train.ipynb` / `*_inference.ipynb`: 未変更。
  2026-07-30の実行承認後にtrainだけ採用し、inferenceは未変更。
- exp408/411/440/441合同pytestは`53 passed / 1 failed`。失敗は実行中の
  exp440 config statusが`kaggle_v1_running_stage0`へ更新済みなのに、
  exp440 testが旧`stage0_authorized_pending_run`を期待している既存不整合で、
  exp441 assertionではない。この依頼では実行中exp440を変更しない。

## 設計契約

- 科学差分はrate kernelだけ。
- 親Euler tri-diagonalを、同じ`momentum`/`sig_r`から導くbin-integrated exact OUへ置換。
- finite rate support外のmassは親と同様に捨て、端で再正規化しない。
- Stage 0は1 variant×32 candidate HMM well-runs、parent rerun 0。
- Stage 1は全gate PASSと別承認時のみ1 variant×773 wells。
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`。
- 同一OOFで`sig_r`、`momentum`、support、emission、gateを救済しない。

## 再現性メモ

- RNGなし。well、row、position、source/destination rate、reduction順を固定する。
- kernelはfloat64で事前計算し、analytic in-support mass・平均・分散を監査する。
- fixed32、episode/cause、保存controlのSHAをconfig固定した。
- transition kernel、prediction、diagnostic、metricsのlogical/content SHAを保存する。
- 採用候補のKaggle train kernel IDは
  `kentookumura/exp441-full-support-ou-rate-transition-hmm-train`に固定した。
- 初回runをdeterministic anchorとせず、独立rerun一致後だけ再判定する。

## 実装内容

- `kappa=-log(momentum)`、`decay=exp(-kappa*delta_MD)`、
  exact OU varianceをfloat64で実装した。
- 41 rate centerの有限Voronoi edgeを作り、各binへGaussian CDF差を
  積分する。support外tailは捨て、transition rowを再正規化しない。
- rate kernelを先に適用し、destination rateで
  `r_destination*delta_MD-delta_Z`の親5点position kernelを適用する。
- Gaussian GR emission、prefix population std、position/rate grid、prior、
  forward/backward、posterior mean/stdをexp209に固定した。
- exact OU kernelはfloat64、alpha/beta messageは親と同じfloat32とした。
- predictive / filtered / smoothed rate mean、rate std、edge massを
  target-free diagnosticとして保存する。
- analytic in-support mass、OU mean/variance、position kernel parity、
  小規模dense HMM brute-forceをtechnical contractにした。
- 全32 wellのtransition kernel / prediction / diagnostic SHAをfreeze後だけ、
  role/fold、truth、persistent episode、cause、exp408 parent row ledgerを読む。
- zero-directed under-responseは、true rateが非zero、decoded rateが同方向または
  zero、かつ絶対値がtrue rateより小さい行として固定した。
- exp408 parent row ledgerのdecompressed SHA
  `74bb3c6b5593c3e01065b9feb81d4f76ee5133eef67a8e8972df22eb61ad2ffb`
  を固定し、parent/candidateのunder-response SSE shareを同じepisode rowで比較する。

## 実行量契約

- Stage 0:
  - scientific candidate: 1
  - candidate HMM well-runs: 32
  - parent control rerun: 0
  - reporting folds: 5
- Stage 1（Stage 0全PASS・別承認時のみ）:
  - scientific candidate: 1
  - candidate HMM well-runs: 773
  - parent control rerun: 0
- LightGBM config / trained fold / booster / fitted model / PF / Beam / GPU:
  `0 / 0 / 0 / 0 / 0 / 0 / 0`

## 次のアクション

1. exp441はterminal closeとし、Stage 1、rerun、inference、submissionへ進まない。
2. exp442は「exp441がtechnical/control-safeかつ方向正・量不足」の先行条件を
   満たさないため、現行設計のStage 0を実行しない。
3. 失敗原因を追加確認する場合だけ、保存済みrate diagnosticを使う
   0-HMM / 0-predictionのtruth-late attribution readoutを別実験・別承認で行う。
