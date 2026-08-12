# exp337_prefix_backtested_structure_sigma_gr セッションノート

## 目的

finite-only `sigma_GR`へknown-prefix backtest由来の構造分散を加え、zero-fillが偶然担っていたGR emissionの過信防止を明示的な不確実性分解へ置き換える設計を確定する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU Stage 0 version 1 FAIL / 枝を終了
- CV / LB / submission: なし
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 失敗根拠: `exp307_finite_only_robust_sigma_gr`
- 実行量: Stage 0は1 diagnostic。HMM/model/LightGBM/fold/PF/Beam/booster/control再実行はすべて0。Stage 1は未実装・未実行。

## コマンドログ

```bash
make new-steering EXP=exp337_prefix_backtested_structure_sigma_gr
make new-exp EXP=exp337_prefix_backtested_structure_sigma_gr
```

- 2026-07-22: steeringとexperiment scaffoldを作成した。
- 2026-07-22: design-only文書、config、バックログを具体化した。
- design-only時点では実装、Notebook編集、validation、Kaggle package/push/run、inference、submissionは未実施だった。

### 設計・実装時点の検証コマンド

```bash
make validate-exp EXP=exp337_prefix_backtested_structure_sigma_gr
make validate-template
```

## 変更点

- 科学的変更は`gr_sigma`だけ。`sigma_eff^2=sigma_finite^2+tau_structure^2`に固定した。
- known-prefix内部splitを60%/40%、rolling originを60%/80%、forward blockを20%に固定した。
- finite pair合計50、early/late各20を最低条件とし、不足時は同prefixのexp209 zero-fill scaleへfallbackする。
- Stage 0でfinite-only/zero-fillとのforward NLL比較を通過した場合だけStage 1を許可する。
- Gaussian emission center、HMM grid/transition/prior/output、LikPF blend weightは固定する。
- MAD、affine、temperature、heavy-tail、row-wise scale、missing downweight、transition、blend救済を禁止する。

## 再現性メモ

- seed policy: RNGなし。well/row/origin/scale policy固定順。
- stochastic components: なし。PF/LikPF/Beamの新規生成なし。
- CPU/GPU runtime: Kaggle CPU、GPU/internet off、`143.899363 sec`。
- Kaggle kernel: `kentookumura/exp337-prefix-backtested-structure-sigma-gr-train` version 1、id_no `128220965`。
- scientific contract SHA: `57fa5c9e3def170f8a3a83018eb4d69ab69ef835f5b61511633c066189feddb5`。
- input dependency contract SHA: `7f19db5ec37b524b9caf17beeee30b281f11760fb6061a1dfe4a3bccc9cbef32`。
- rolling-origin audit content SHA: `3f72fb1dcb4ea95c4b77b54d2b75c7f302dd0fd7fbd2707ddcd2c5532dd0e883`。
- full-prefix audit content SHA: `b83a5a6c41dc6887afec2840c160482091aafcf26fd19de5f5167487888fd2b8`。
- model manifest / model SHA: 非該当。
- prediction SHA: 非該当。Stage 1を実行していない。
- submission SHA: 非該当。inference/submission禁止。
- rerun check: version 1のKaggle logsとoutput file listを確認。output archiveは取得していない。

## 次のアクション

- なし。固定Stage 0 gate FAILによりこの枝は終了し、同一結果上の救済を行わない。

## 2026-07-22 Stage 0実装

- ユーザーの「exp337を実装してください」を、Stage 0 compact self-contained train候補、fail-closed inference候補、専用test、設定・記録の実装承認として扱った。Kaggle package/push/run、Stage 1 HMM、inference、submissionは範囲外。
- `GR`と`TVT_input`だけを選択するhorizontal loader、exp209互換のType Well TVT sort + GR ffill/bfill + linear interpolationを実装した。unknown suffix `TVT`、error、oracle列はStage 0で読まない。
- origin `0.60/0.80`の直後20% blockをforward評価に固定した。fitはorigin以前だけを使い、その有限residualを60/40分割してearly population stdとlate zero-center MSEから`tau_structure`を計算する。
- total finite 50未満、early/late各20未満、nonfinite時は、同じ利用可能prefix上のexp209 zero-fill stdへexact no-op fallbackする。finite-only comparatorへlate residual stdを二重計上しない。
- 両originのevaluable/fallback/pooled NLL/fold NLL、zero-fill比gain、full-prefix median tau/lower clipを固定AND gateにした。PASS時もStage 1 flagはfalseのまま。
- exp209 raw HMM/exp072 cache、exp226 fold、exp115 hidden-like、exp307 negative scale/summary、raw well identityのSHA/column/coverage preflightを高コスト処理前に行う。
- 生成物はscientific contract、input dependency contract、well×origin scale/NLL、full-prefix scale、Stage 0 gate、summary。gzipはdecompressed content SHAを主証拠にする。
- 正規placeholder Notebookは上書きせず、`*_compact_selfcontained_train.py/.ipynb`と`*_compact_selfcontained_inference.py/.ipynb`を作成した。

### 実行量ガード

- Stage 0 diagnostic: 1
- HMM / model / LightGBM config / trained fold / PF / Beam / booster: `0 / 0 / 0 / 0 / 0 / 0 / 0`
- 親control再実行: 0
- Stage 1: 未実装、未承認

### 実装検証

```text
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb <compact_train.py> <compact_inference.py>
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test <compact_train.py> <compact_inference.py>
.venv/bin/python -m py_compile <compact_train.py> <compact_inference.py> experiments/exp337_prefix_backtested_structure_sigma_gr/tests/test_exp337_prefix_backtested_structure_sigma_gr.py
.venv/bin/ruff check <compact_train.py> <compact_inference.py> experiments/exp337_prefix_backtested_structure_sigma_gr/tests/test_exp337_prefix_backtested_structure_sigma_gr.py --select E,F,I,UP,B
.venv/bin/pytest -q experiments/exp337_prefix_backtested_structure_sigma_gr/tests/test_exp337_prefix_backtested_structure_sigma_gr.py
make validate-exp EXP=exp337_prefix_backtested_structure_sigma_gr
make validate-template
```

- 専用test: `10 passed`。
- Jupytext round-trip、py_compile、Ruff: PASS。
- strict experiment validation、template validation: PASS。
- 全体pytest: `568 passed, 3 skipped, 2 failed`。2 failureは既存exp296の完了済みstatus / disabled run flagに対して旧実行前状態を期待する既知不整合で、exp337専用testとshared Notebook testは全PASS。
- 親exp209にcompact sourceはない。正規Jupytext sourceはexp209 `174行/6章`、exp337 compact trainは`1,104行/9章`で、helper importだけの薄いNotebookではない。
- 実装時点では`__file__`参照0、正規Notebook上書き0、Kaggle package/push/run 0。その後の明示実行承認に基づき正規Notebookへ採用した。

## 2026-07-22 Kaggle CPU Stage 0実行承認

- ユーザーの「実行してください」をStage 0 Kaggle CPU package/push/runの明示承認として記録した。Stage 1 HMM、inference、submissionは未承認。
- 実行量: 1 diagnostic、HMM/model/LightGBM config/trained fold/PF/Beam/booster/control再実行は`0/0/0/0/0/0/0/0`。
- compact self-contained train候補を正規train Notebookへ採用した。正規Notebookの置換はこの実行承認に基づく。
- credential check: OAuth PASS、legacy Kaggle key PASS、CLI `2.2.3`。API tokenは未設定だがCLI OAuth/legacy経路を使う。
- input kernel source 4件を`kaggle kernels pull -m`で確認した: exp209 `id_no=126193687`、exp226 `126463591`、exp115 `124519917`、exp307 `128085112`。すべてCPU/internet off。
- canonical kernel id/title: `kentookumura/exp337-prefix-backtested-structure-sigma-gr-train` / `exp337 prefix backtested structure sigma gr train`。
- package前のJupytext、py_compile、Ruff、専用+shared Notebook test `14 passed`、strict experiment/template validationはPASS。

### package / push / 初期状態

```text
make prepare-kaggle-notebooks EXP=exp337_prefix_backtested_structure_sigma_gr EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp337-prefix-backtested-structure-sigma-gr-train --title 'exp337 prefix backtested structure sigma gr train' --run-on-push --strict"
kaggle kernels push -p experiments/exp337_prefix_backtested_structure_sigma_gr/kaggle/train -t 300
kaggle kernels pull kentookumura/exp337-prefix-backtested-structure-sigma-gr-train -p /tmp/exp337-kaggle-pull-v1 -m
kaggle kernels status kentookumura/exp337-prefix-backtested-structure-sigma-gr-train
kaggle kernels logs kentookumura/exp337-prefix-backtested-structure-sigma-gr-train
```

- package metadata/bootstrap: canonical id/title、private、CPU、internet off、competition source 1件、kernel sources 4件、Stage 0 diagnostic 1、HMM/model/booster/control再実行0、Stage 1/inference/submission falseを確認した。
- push: `Kernel version 1 successfully pushed`。
- pulled metadata: canonical id/title一致、`id_no=128220965`、CPU/internet off、kernel sources 4件一致。
- initial status: `KernelWorkerStatus.RUNNING`。初期logsは空で、実行中logsが返らない既知挙動として同一versionを監視する。

## 2026-07-22 Kaggle CPU Stage 0完了

- status: `KernelWorkerStatus.COMPLETE`。Notebook summary runtimeは`143.899363 sec`。
- 773 wells、2 origins、1,546 rolling rowsを評価した。両originのevaluable coverageは`773/773 = 1.0`、fallbackは`0/773 = 0.0`。
- origin 0.60は186,184 finite pairs。pooled NLLはfinite-only `3.027164870115609`、zero-fill `3.589239068245176`、structure-added `3.0738661030667833`。structure-addedはzero-fill比`0.5153729651783925/residual`改善・5/5 folds勝利だが、finite-only比0/5 foldsでFAIL。
- origin 0.80は178,469 finite pairs。pooled NLLはfinite-only `2.9718541607003055`、zero-fill `3.571888954744798`、structure-added `3.0157835169957905`。structure-addedはzero-fill比`0.5561054377490073/residual`改善・5/5 folds勝利だが、finite-only比0/5 foldsでFAIL。
- full-prefixは773 wells、fallback 0、median `tau_structure=0.0`で最低`5.0` gateをFAIL。lower clipは`42/773 = 0.054333764553686936`で上限0.10以内。
- unknown-suffix truthのfreeze前readはfalse。HMM/model/LightGBM/fold/PF/Beam/booster/control再実行はすべて0。
- 主成果物として`rolling_origin_scale_nll_audit.csv.gz`、`full_prefix_scale_audit.csv.gz`、`stage0_gate_summary.json`、`scientific_contract.json`、`input_dependency_contract.json`、`summary.json`、`metrics.json`をoutput file listで確認した。大きなoutput archiveは取得していない。
- final statusは`stage0_gate_failed_branch_closed`、decisionは`stage0_fail_close_without_hmm_rescue_inference_or_submission`。structure-addedはzero-fillを改善したがfinite-onlyを下回り、典型wellの構造分散も0だったため、追加構造分散仮説は支持されない。
- 事前固定どおりsplit/threshold/scale/likelihood救済をせず、Stage 1 HMM、inference、submissionへ進まない。
