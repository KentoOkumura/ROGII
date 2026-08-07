# exp262 セッションノート

## 2026-07-17 Kaggle train v1完了・selector guard不通過

canonical `kentookumura/exp262-sel-extra-trees-exp238-train` version 1 / `id_no=127468598` は`COMPLETE`。CPU / internet offで、20 selector boostersを完走した。ログ最大時刻は18,760.043秒（約5時間13分）。Traceback、OOM、実行errorはない。

### 実行契約

- active variant 1 (`extra_trees_true`) × selector config 1 × outer 5 × inner 4 = 20 CPU boosters。
- historical exp238 control再学習0、downstream exp218再学習0、inference 0、submission 0。
- 3,783,989 rows / 773 wells、11 candidates、184 context + candidate-long 3 = 187 selector features。
- model manifestは20/20、outer×inner被覆完全、全modelで`extra_trees=True`。best iterationはmean 1,172.4 / median 1,199 / min 914 / max 1,200。

### historical exp238との差

| 指標 | historical | extra-trees | 差 |
| --- | ---: | ---: | ---: |
| candidate error MAE | 4.532912 | 4.748342 | +0.215430 |
| oracle-candidate logloss | 2.957410 | 3.025668 | +0.068258 |
| pairwise rank accuracy | 0.792762 | 0.783041 | -0.009721 |
| fixed top1 global RMSE | 8.512262 | 8.840332 | +0.328071 |
| fixed Viterbi global RMSE | 8.492559 | 8.826521 | +0.333962 |
| fixed Viterbi 000-050 RMSE | 0.667616 | 0.627142 | -0.040474 |
| fixed Viterbi 1000+ RMSE | 9.314887 | 9.694635 | +0.379748 |
| fixed Viterbi exp115 spatial RMSE | 8.821805 | 9.620149 | +0.798344 |
| fixed Viterbi exp115 typewell-purged RMSE | 8.778809 | 9.556358 | +0.777549 |

- score surfaceは変化した（flat Pearson 0.998484、rowwise Spearman 0.897034、top1 agreement 0.429939、mean absolute score difference 1.493698）。
- fixed Viterbi fold差は`+0.677187 / +0.258589 / -0.092081 / +0.219028 / +0.615652`で、nonworse 1/5 folds。
- current-vs-historical最大well回帰は`a959858c` fixed Viterbiの+12.835997 ft。known worst vs likpfはhistorical +37.680899に対しcurrent top1 +37.374155で拡大しなかった。
- guardはnear非悪化とknown-worst非拡大だけを含む3/12 checks PASS、全体はFAIL。

### output / SHA監査

model manifest、5 nested score gzip、fixed selection prediction、candidate/selection/by-well metrics、feature importance、guard、summaryをKaggle outputから`/tmp`へ取得して実ファイル監査した。

- summary記録の主要11 files raw SHA: 11/11一致。
- nested scores: 5/5 raw SHAとdecompressed SHAが一致。
- selector models: 20/20 SHA一致。
- fixed selection prediction decompressed SHA: `ac3cc0bfa8d132641453f7f404f829ef3d9e2abc63e7151270b8a89c2363f41a`、一致。
- selector model manifest SHA: `5e0dbc5f3f1e6b76b214e24a3e523f1aaff03019f82b8fd20e2c7c51ec851807`。
- selector feature content SHA: `096b6788fd915b98de4d4a3035f274415ddc8661f6d0021f8714902bfff6299d`。
- rerun SHA一致は未確認のためdeterministic anchorとは扱わない。

### 判断

score差は十分あるが、candidate error calibration、ranking、global、1000+、hidden-like、4/5 folds、worst-well stabilityが同時に悪化した。実装/SHA/fold契約は通っているため、原因は実装失敗ではなく、この固定surfaceでのrandom-threshold extra-trees仮説が候補誤差scoreの汎化を損ねたことと判断する。

selector extra-trees branchは不採用として閉じる。tree数、leaf、sampling、temperature、thresholdの救済grid、downstream 15 boosters、raw-test inference、submitは行わない。既存高優先の`exp264_exp263_candidate_confidence_dual_selector` Stage Bはstandard LightGBM固定であり、本結果に依存せずextra-treesを避ける次ルートとして維持する。新規backlogは追加しない。

## 2026-07-16 Kaggle train v1 実行承認

ユーザーの「実行してください」を、次の限定scopeに対する明示承認として記録した。

- active variant: 1 (`extra_trees_true`)
- selector config: 1
- folds: outer 5 × inner 4
- 合計: 20 CPU boosters
- historical exp238 control再学習: 0
- downstream exp218 LightGBM再学習: 0
- GPU booster: 0
- approved scope: `extra_trees_true_1_variant_1_selector_config_outer5_inner4_20_cpu_boosters_no_control_or_downstream_retraining`
- canonical kernel: `kentookumura/exp262-sel-extra-trees-exp238-train`
- Kaggle metadata: private / CPU / internet off

この承認はselector train v1だけに適用する。guard通過後のdownstream再学習、raw-test inference、competition submitは別承認とし、今回実行しない。

### push前package監査

- `make validate-exp EXP=exp262_selector_lightgbm_extra_trees_ablation_on_exp238`: strict PASS。
- `make validate-template`: PASS。
- `make test`: 50 tests PASS。
- `make prepare-kaggle-notebooks ... --run-on-push --no-src --strict`: PASS。
- canonical id/title: `kentookumura/exp262-sel-extra-trees-exp238-train` / `exp262 sel extra trees exp238 train`。
- metadata: private / CPU / internet off / `run_on_push=true` / kernel sources 9件。
- bootstrap: support 10 files。埋め込みconfigは`run_approved=true`、approved/required scope一致、20 boosters、control/downstream再学習0、`extra_trees=True`を確認した。
- push前の`kaggle kernels pull`は未作成kernelに対する`GetKernel` 403。slugを変えずに初回pushした。
- `make push-kaggle-train EXP=exp262_selector_lightgbm_extra_trees_ablation_on_exp238`: canonical kernel version 1のpush成功。Kaggle train v1を実行中。
- URL: `https://www.kaggle.com/code/kentookumura/exp262-sel-extra-trees-exp238-train`
- push後の`kaggle kernels pull ... -m`: PASS。`id_no=127468598`、private CPU / internet off / 9 kernel sourcesをKaggle側metadataで確認した。
- 定期監視では20:22 JSTまで`RUNNING`を確認し、実行中のCLI logsは空だった。ユーザー指示により定期監視だけ停止した。Kaggle kernel version 1自体は停止・cancelしていない。完了連絡後に同一kernelのlogsと必要な生成物を確認する。

## 2026-07-16 実装・静的検証

`selector_lightgbm_extra_trees_ablation_on_exp238` を単一parameter ablationとして
`exp262_selector_lightgbm_extra_trees_ablation_on_exp238` に切り出した。

### Kaggle train前コスト

- active variant: 1 (`extra_trees_true`)
- selector config: 1
- folds: outer 5 × inner 4
- 合計: 20 CPU boosters
- historical exp238 control再学習: 0
- downstream exp218 LightGBM再学習: 0
- GPU booster: 0
- Kaggle push: 静的検証後もユーザーの明示承認まで禁止

### 固定契約

- exp238 train v4の11候補、184 context + candidate-long 3列。
- well GroupKFold outer 5 × inner 4、seed 42。
- train/early-stoppingのcandidate-long上限各120,000 rows。
- `regression_l1`、その他LightGBM parameter、fixed Viterbi rule。
- historical exp238の20 selector models、nested OOF score、metricsを保存controlとして使う。

### 変更点

- selector LightGBMへ `extra_trees=True` だけを追加する。
- parameter差分が1個だけであることをruntime assertionする。

### 実装内容

- Jupytext percent形式のself-contained train sourceを1,335行 / 7章で作成した。
- 親exp238にcompact self-contained selector trainはない。親の正規selector trainは182行 / 7章で同一exp helperへ主要処理を委譲していた。exp262は重いexp237 candidate feature builder / fixed Viterbiだけを親sourceとして再利用し、cost、parameter差分、candidate surface、fold、historical input、nested学習、評価、guard、保存をnotebook上へ展開した。
- historical summary / model manifest / nested scoresをrow、id、well、role、outer fold、candidate列、184 context列、187 selector入力列、20 modelsで検証する。
- canonical exp238 configをbootstrapし、selector params、seed、outer/inner fold、学習行上限、chunk、20 booster契約が完全一致することを学習前にassertする。
- candidate score calibration/ranking、score correlation、fixed top1/Viterbi、global / near / 1000+ / hidden-like / fold / by-well / worst-well guardを実装した。
- feature importance mean/top plot、model manifest、nested scores、fixed selection prediction、各SHAを生成する。
- inference notebookはselector guard通過と別途承認まで意図的に停止する。

### コマンドログ

- `make new-steering EXP=exp262_selector_lightgbm_extra_trees_ablation_on_exp238`: steering作成。
- `make new-exp EXP=exp262_selector_lightgbm_extra_trees_ablation_on_exp238`: 実験作成。
- train/inferenceの`jupytext --to ipynb`と`jupytext --to ipynb --test`: PASS。
- train/inferenceの`py_compile`: PASS。
- train/inferenceの`ruff --select F821`: PASS。
- train/inferenceのfull `ruff check`: PASS（B905を`zip(..., strict=True)`で修正後に再検証）。
- `make validate-exp EXP=exp262_selector_lightgbm_extra_trees_ablation_on_exp238`: strict PASS。
- `make validate-template`: PASS。
- `make test`: 50 tests PASS。
- canonical local package `kentookumura/exp262-sel-extra-trees-exp238-train`を`run_on_push=false`、CPU、internet off、kernel sources 9件、bootstrap support 10 files（exp238 config 1件 + exp237依存4件）でstrict生成した。`--no-src`で未使用のrepo `src/`を除き、trainで未使用のexp237 raw-test helperも同梱していない。
- 埋め込みconfigはactive variant 1、selector config 1、outer 5 × inner 4、20 CPU boosters、control/downstream再学習0、`extra_trees=True`だけ、`run_approved=false`、`approved_scope=null`であることを確認した。
- ローカルnotebook実行・学習は行っていない。初回実行はKaggle CPU Notebookとする。

### 再現性

- bounded samplingはmodelごとのlocal `np.random.default_rng` とexp238 seed式を維持する。
- LightGBM random stateはouter/inner foldから固定する。
- CPU実行だが、同一kernel rerunでmodel/prediction SHA一致を確認するまではdeterministic anchorと扱わない。
- context feature content SHA、schema SHA、historical/new nested score decompressed SHA、20 model SHAを保存する。
- PF/Beam/HMM candidateを新規生成する実験ではなく、固定candidate surfaceを入力にする。

### 次

1. `run_approved=true`と承認scopeを埋め込んだ`run_on_push=true` packageを再生成し、CPU / internet off / kernel sources 9件 / bootstrap support 10 files /埋め込みconfigを再監査する。
2. canonical Kaggle CPU train v1をpushし、同一kernel idの存在と実行完了を監視する。
3. train完了後にCV、guard、runtime、生成物path、kernel version、SHAを記録する。
