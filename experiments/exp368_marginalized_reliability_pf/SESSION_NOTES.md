# exp368_marginalized_reliability_pf セッションノート

## 目的

exp072 likelihood-PFの各粒子でsticky `normal / weak` GR reliabilityをsampleせず
Rao-Blackwell化する前に、追加PFを回さない固定Stage 0で識別可能性を監査する。

## 現在の状態

- Route: `pf_beam`
- 状態: `stage_0_failed_close_without_rescue`
- CV / LB: なし
- Notebook: compact self-contained trainとfail-closed inferenceを実装し、
  placeholderの正規Notebookを置換済み
- Kaggle package / push / run: canonical private CPU kernel v1完了
- Stage 1 PF / inference / submission: 不適格・未実装・未実施

## コマンドログ

### 2026-07-23 scaffold作成

```bash
make new-steering EXP=exp368_marginalized_reliability_pf
make new-exp EXP=exp368_marginalized_reliability_pf
```

### 2026-07-25 Stage 0実装

ユーザーの`exp368を実装してください`を、既存steeringにあるStage 0実装と
placeholder Notebook置換の承認として記録した。Kaggle実行、Stage 1実装、推論、
提出の承認には拡張していない。

実装内容:

- known-prefix preflight:
  - horizontalから`GR / TVT_input`、Type Wellから`TVT / GR`だけを読む。
  - finite `TVT_input`の最終連続192行を、128行history + 64行held-outに固定する。
  - GRはexp072と同じboth-direction interpolation + Type Well平均fallbackを使う。
  - sigmaはexp072と同じ全known prefix残差（missing raw GRは0埋め）から作り、
    `[10,60]`へclipする。
  - q posteriorをhistoryからheld-outへ持ち越し、base sigma 1倍Gaussianと
    sigma 1倍/4倍marginalized Gaussianの逐次予測NLLを比較する。
- saved exp072 suffix readout:
  - SHA固定cacheから`id / well / last_known_tvt / likpf_mean_d`だけをparseし、
    cache内`target`は読まない。
  - `last_known_tvt + likpf_mean_d`を固定pathとする。
  - sigmaはexp072と同じ全known-prefix残差標準偏差を使い、missing known GRは0埋めする。
  - suffix GRはboth-direction interpolation後Type Well平均fallbackとする。
  - 512行 / stride 256で短い末尾blockも保持し、各blockでqを`[0.8,0.2]`へ戻す。
  - weak posterior block平均をfreezeし、well内SHA256由来nonzero circular shiftを
    negative controlにする。
- truth-late gate:
  - known-prefix NLLとsuffix block ledger / weak posteriorをdeterministic gzipへ保存し、
    decompressed content SHAを確定してからexp226 truth/foldとexp115 hidden-like roleを読む。
  - pooled NLL gain、bad10 AUC、circular差、4/5 folds、hidden-like 2面、
    weak massを固定AND gateで評価する。
- Stage 1の粒子replay、alpha ancestor copy、PF resampling、prediction生成は
  Stage 0 PASSと別承認前なのでコードに含めていない。

## 実行コスト契約

- Stage 0 diagnostic variant: 1
- reporting folds: 5
- PF seed-well runs / PF control replay: `0 / 0`
- model config / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- 親exp072 PF再実行・再学習: 0
- Stage 1予約: 全gate PASSかつ別承認時だけ
  1 scientific variant / 1 treatment replay / 0 control replay /
  500 particles / 128 seeds / 773 wells / 98,944 seed-well runs

## 再現性メモ

- Stage 0はRNGなし、CPU single worker。
- q recursionはnormalized Gaussian log-densityをlog-sum-expで周辺化する。
- target-free prefix/suffix生成物のdecompressed content SHAをtruth join前に固定する。
- saved exp072 cacheはraw gzip SHA
  `14faee3a24de587b7190e7febffd33cc1256e418c7505f2d1e6f5f7c9b3c2f18`
  をhard guardする。
- 将来のStage 1は
  `SHA256(experiment|well|family|seed_index)`のlocal RNGを必須とし、
  global RNG / thread schedule依存を禁止する。
- kernel / output SHA: version 1完了後に下記へ記録。
- prediction / model / submission SHA: 非該当・未生成。
- raw input preflightで全773 wellの最終連続known `TVT_input`長を確認し、
  最小851行、192行未満0 wellだった。`GR`自体は疎に欠損するため、親exp072と異なる
  finite-GR連続条件は採用していない。

## 実装検証

```bash
.venv/bin/python -m py_compile \
  experiments/exp368_marginalized_reliability_pf/exp368_marginalized_reliability_pf_compact_selfcontained_train.py \
  experiments/exp368_marginalized_reliability_pf/exp368_marginalized_reliability_pf_compact_selfcontained_inference.py \
  tests/test_exp368_marginalized_reliability_pf.py
.venv/bin/ruff check \
  experiments/exp368_marginalized_reliability_pf/exp368_marginalized_reliability_pf_compact_selfcontained_train.py \
  experiments/exp368_marginalized_reliability_pf/exp368_marginalized_reliability_pf_compact_selfcontained_inference.py \
  tests/test_exp368_marginalized_reliability_pf.py
.venv/bin/pytest -q tests/test_exp368_marginalized_reliability_pf.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp368_marginalized_reliability_pf/exp368_marginalized_reliability_pf_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp368_marginalized_reliability_pf/exp368_marginalized_reliability_pf_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb \
  experiments/exp368_marginalized_reliability_pf/exp368_marginalized_reliability_pf_compact_selfcontained_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test \
  experiments/exp368_marginalized_reliability_pf/exp368_marginalized_reliability_pf_compact_selfcontained_inference.py
.venv/bin/python scripts/validate_experiment.py \
  --experiment exp368_marginalized_reliability_pf
.venv/bin/python .agents/skills/kaggle-review-exp/scripts/review_exp_docs.py \
  exp368_marginalized_reliability_pf --root .
```

- `py_compile`: PASS
- `ruff`: PASS
- 専用test: `11 passed`
- Jupytext変換 / round-trip: PASS
- `validate-exp`: PASS
- Kaggle / local Notebook実行: なし

### 2026-07-25 Stage 0 Kaggle実行承認

ユーザーの`実行してください`を、固定済みStage 0をcanonical private CPU kernel
`kentookumura/exp368-marginalized-reliability-pf-train`へpackage / pushし、
完了まで監視する承認として記録した。

- 実行対象: Stage 0 diagnostic 1 variant / reporting 5 folds
- PF seed-well run / PF control replay: `0 / 0`
- model config / LightGBM config / trained fold / booster: `0 / 0 / 0 / 0`
- 親exp072 control再実行: 0
- accelerator / internet: CPU / disabled
- Stage 1 PF / inference / submission: 未承認、無効のまま
- 実行フラグ: `run_stage_0=true`、`run_stage_1=false`、
  `run_inference=false`、`create_submission=false`

実行記録:

```bash
task prepare-kaggle-notebooks ...  # task未導入のためcommand not found
make prepare-kaggle-notebooks EXP=exp368_marginalized_reliability_pf \
  EXTRA_ARGS="--notebook train \
  --kernel-id kentookumura/exp368-marginalized-reliability-pf-train \
  --title 'exp368 marginalized reliability pf train' --run-on-push --strict"
kaggle kernels push \
  -p experiments/exp368_marginalized_reliability_pf/kaggle/train
kaggle kernels pull \
  kentookumura/exp368-marginalized-reliability-pf-train \
  -p /tmp/exp368-kaggle-pull-v1.38k4KF -m
```

- push: version 1成功
- Kaggle kernel id: `kentookumura/exp368-marginalized-reliability-pf-train`
- Kaggle id_no: `128591117`
- Kaggle metadata: private / CPU / internet off / competition source 1 /
  kernel source 3
- 初回監視: 実行中のためCLI logsは空。空ログを理由に再pushしない。

### 2026-07-25 Stage 0 Kaggle CPU version 1完了

- 最終status: `COMPLETE`
- runtime: `630.531264 sec`
- 3,783,989 rows / 773 wells / 15,174 suffix blocks /
  49,472 known-prefix held-out rows
- technical gate: PASS
  - 全score finite、expected rows / wells / folds一致
  - known-prefix 64 rows × 773 wells
  - truth columns read before freeze 0
  - strict quartile、multi-block circular offset nonzero
  - PF / model / booster / parent control rerunはすべて0
- scientific gate: FAIL
  - known-prefix NLL gain:
    `0.000373564 < 0.01`でFAIL
  - pooled bad10 AUC:
    `0.636675 >= 0.60`でPASS
  - real - circular AUC:
    `0.058264 >= 0.02`でPASS
  - real AUC > 0.50 folds: `5/5`でPASS
  - hidden-like spatial / typewell-purged:
    `0.641795 / 0.636115 >= 0.55`でPASS
  - row-weighted weak mass:
    `0.009689 < 0.02`でFAIL
- decision: `stage_0_failed_close_without_rescue`
- Stage 1 eligibility: false
- 同一OOFでのtransition、sigma multiplier、block、threshold、gate、blend救済、
  再push、Stage 1、inference、submissionは行わない。
- gate / SHAの実ファイル確認が必要なためKaggle output 6.7 MBを
  `kaggle/output/train_v1`へ取得した。

fold別real bad10 AUC:

- fold 0: `0.627147`
- fold 1: `0.627234`
- fold 2: `0.633846`
- fold 3: `0.653398`
- fold 4: `0.639133`

再現性SHA:

- scientific contract content:
  `dd333d921e377447f1f4c1c49c77bd852122eab059e1532ba9a2f22013ba1314`
- block ledger content:
  `7327ce8e6383d76f99c51cec6982c1db181e6f05257df28e7268d7a0549ba30a`
- known-prefix NLL content:
  `eeb5d7981a8926753a20435b6b816eeb6548877f0171be18f3113069f07a2811`
- weak posterior content:
  `4ffa4fc761fc4db6b1c7de42c132b8102e33f9910bf5dc56752b20e95c2520ae`
- late-truth block readout content:
  `5f90ed658c09c2dc54f52a617f1f2467c46939cd7c955828463129bc7d611189`
- gate raw:
  `bb2e83cbcecdefa9c195987f18d1fa3b58d81bcf33f1f11c6f2ec21dd5d53e48`
- downloaded summary raw:
  `fcf0a17d31ae242fb6bf74bfdf333152ad40c30b22bbe9ab14bc63bf1a7650ae`
- downloaded metrics raw:
  `996347540307b814db026b09f817b1ec918155ddc680a5a906de2fa6ddf9a4b3`
- kernel log raw:
  `c83c970ba258378c3d58e55ee44316ca31ea6df4139b47ba0bc3ab5967969197`

親exp072にはcompact self-contained train sourceがないため、同形式の行数比較は不可。
exp368正規train Notebookは1,678行 / 21 cellsで、Imports、runtime/config/SHA、
scientific contract、target-free exp072 path、known-prefix NLL、suffix freeze、
late truth、metrics/gates、orchestrationをNotebook内に展開している。
同じ実験ディレクトリのhelper importはない。

## 次のアクション

1. exp368 branchを閉じる。
2. Stage 1 PF、inference、submissionは実装・実行しない。
3. 同じqの調整救済は行わない。再訪にはknown-prefixとsaved suffixのweak
   activation乖離を説明する独立truth-free監査を要求する。
