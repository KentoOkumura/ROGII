# exp503 セッションノート

## 目的

exp490が強い/弱いwellをtruth-awareに多面的分解し、公開notebook型のprefix fadeと
prefix条件付き補正量が外側foldでも有効かを監査する。

## 現在の状態

- Route: `ensemble`
- Status: `completed_diagnostic_no_inference`
- Kaggle: id_no `129477630` / version `3` COMPLETE
- exp490 / tau500 / prefix tree RMSE: `8.480155260 / 8.447032560 / 7.911444631`
- parent: exp490、fallback: 保存済みexp357
- inference/submission: 無効

## 固定設計

- 入力: exp490 full OOF 3,783,989 rows / 773 wells、exp499 target-free 32 features。
- truth-aware: well誤差形状、depth、feature、archetype、代表軌跡。
- fold-safe: 29 fade profilesのouter-4選択、prefix/context alpha tree。
- nondeployable: early 128/256/512 truthでprofileを選び後半へtransferする楽観監査。

## Push前の実行量確認

- variants: 1
- fixed fade profiles: 29
- outer folds: 5
- alpha tree feature sets / fits: 2 / 最大10
- KMeans fits: 1
- LightGBM configs / boosters: 0 / 0
- new PF / HMM / Beam / candidate predictions: 0 / 0 / 0 / 0
- parent/control retraining: 0
- GPU runs: 0

保存済みOOFだけを使うCPU readoutで、control再学習を含まない。

## 再現性

- seed: 42、KMeans `n_init=20`、tree `random_state=42`。
- single process、parallel RNGなし。
- exp490 raw gzip SHA `99030b33...61b72c`、decompressed SHA
  `e020e82e...e9a07`。
- exp499 feature SHA `54c7e1da...7bb0d4`。
- submission SHA: not applicable。

## 外部/公開notebook根拠

- `fleongg/fle3n-rogii-v5`: `1-exp(-md_since/tau)`、公開設定tau=85のwarm-up damping。
- `curvecowboy/rogii-lb7201-public-gold-conservative`: 既知prefixをmaskして候補を
  backtestし、低alphaで保守的に適用する設計。
- exp503は前者を保存済みOOFで再生する。後者の完全再生に必要なcutoff別HMM再実行は
  行わず、early-truth transferで追加replayの価値だけ判定する。

## コマンドログ

### 2026-08-02

```bash
make new-steering EXP=exp503_exp490_strength_weakness_prefix_policy_readout
make new-exp EXP=exp503_exp490_strength_weakness_prefix_policy_readout
.venv/bin/pytest -q experiments/exp503_exp490_strength_weakness_prefix_policy_readout/tests/test_exp503_contract.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp503_exp490_strength_weakness_prefix_policy_readout/exp503_exp490_strength_weakness_prefix_policy_readout_compact_selfcontained_train.py
```

- contract tests: 初回4/4、rows merge regression追加後5/5 PASS。
- `py_compile`、`ruff --select F821`、train/inference Jupytext round-trip: PASS。
- train 17 cells（markdown 9 / code 8）、inference guard 3 cells。
- exp499 compact trainのSHA/path/config/feature freeze、集計、policy、artifact章を参照し、
  exp503ではreadout/depth/fade/tree/transfer/plotをself-containedに展開した。
- scaffold placeholderを正規train/inference notebookへ採用した。

### Kaggle初回push

- strict package、package内contract tests 4/4、CPU/internet off、source 2件を確認。
- 最初のID/title `exp503-exp490-strength-weakness-prefix-policy-readout-train`
  （59文字）は`SaveKernel 400`。参照元exp490/exp499 kernelはpull成功し、元IDのpullは
  403で未作成を確認した。Kaggle title上限に収めるため、意味を残した48文字の
  `kentookumura/exp503-exp490-strength-prefix-fade-readout-train` / 
  `exp503 exp490 strength prefix fade readout train`へ同じexp内で短縮した。
- 短縮後にpackage再生成・contract再検証し、version 1 push成功。
- URL: `https://www.kaggle.com/code/kentookumura/exp503-exp490-strength-prefix-fade-readout-train`

### Kaggle version 1--3

- version 1: SHA検証と3,783,989 rows読込後、凍結特徴`rows`と再集計`rows`が
  mergeでsuffix化し`KeyError`。科学条件は未評価。
- 修正: well row数の完全一致を検証し、凍結側の重複`rows`だけdrop。専用regression
  testを追加し5/5 PASS。
- version 2: technical PASS、全readout完了。exp490 `8.480155260`、outer global fade
  `8.098662373`、prefix/context tree `7.911444631 / 7.967390614`。strong fadeは
  fold 3とtailをFAIL、early-truth replay triggerもFAIL。
- version 3: 29事前gridを変えず、version 2で5/5 fold改善を確認したtau85/500の
  well tailを追加。tau500は`8.447032560`、5/5 folds、exp490比p95/worst
  `+0.080156 / +1.195616 ft`でcharacterization gate PASS。ただしversion 2結果を
  見た後の探索的選択なので独立validationとは扱わない。
- runtime / peak RSS: `57.844874 sec / 1.549519 GiB`。
- actual execution: 29 profiles / 10 tree fits / 1 KMeans、LightGBM / control retrain /
  prediction / HMM / PF / Beam / GPU各0。

### 主要な科学結果

- exp490 449 wells改善 / 324悪化。positive gain上位10がgross positiveの36.9231%、
  worst10がgross harmの57.8542%を占める。
- correction-required alignmentはbenefit Spearman `0.820838`。weak/strong medianは
  `-0.214 / 0.757`で、posterior不確実性より補正方向が支配的。
- 0--512 suffix rowsはexp490が悪化し、1024+で改善が増える。fold 0は全depth悪化。
- strongest target-free signalは`parent_exp226_abs_mean` AUC `0.591912`。hard router
  には不足。
- prefix treeは平均改善するがfold 3 `+0.620514 ft`、well p95/worst
  `+3.458136/+20.766347 ft`でtail-safeではない。
- early 128/256 truth choiceは後半を悪化、512でもgain `0.021980 ft`、transfer
  Spearman `0.150234`。masked-prefix HMM replayを実行しない。

### Artifact回収

- output: `kaggle/output/train_v3`
- well readout SHA: `d87629a1...f8be2`
- depth metrics SHA: `5b030a19...ed9d`
- feature association SHA: `48268a5b...fcc9`
- policy OOF SHA: `0458a783...fd90`
- fold metrics SHA: `e9b0109e...1f2c`
- model manifest SHA: `ae4d0950...37d8`
- summary SHA: `36c18e8c...1056`
- local metrics / config / canonical train notebook SHA:
  `9a519444...97a5` / `ef49fe38...ae0f` / `456df773...a857`
- submission SHA: not applicable

## 次

adaptive prefix / hard-router / masked-prefix replayは閉じる。tau500は再利用時の
exploratory fixed postprocessとして記録するだけで、exp490 standalone inference / submitへ
進めない。新しい独立holdoutまたは別候補で同じfade効果を検証する必要が生じた場合だけ再開する。
