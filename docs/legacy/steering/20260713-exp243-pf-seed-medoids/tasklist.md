# タスクリスト

## TODO

- なし。

## 進行中

- なし。

## ブロック中

- なし。

## 完了

- `exp243_pf_seed_medoids` steering作成。
- exp072互換stable seed / Gaussian likelihood-PF replay実装。
- weighted trajectory RMSEと決定的BUILD+PAM実装。
- K=3/5/8 medoid candidate、cluster manifest、cluster summary実装。
- exp237 base8 union、row/block/whole-well oracle、unique-best、bucket/hidden-like/by-well readout実装。
- raw-test inference / submission無効化。
- synthetic 3-mode trajectoryでdeterministic K-medoidsを確認。
- train / inference notebookのJupytext変換と`--test`。
- `py_compile`、`ruff --select F821`、`make validate-exp` strict PASS。
- canonical Kaggle train package生成とbootstrap / metadata契約確認。
- 4 shard wrapper作成、Jupytext変換、shard 0 packageのentrypoint/bootstrap確認。
- `experiment_summary.md`更新、`pf_seed_medoids`を未着手backlogから実装済み表へ移動。
- exp072 stable seedの`modulo + 1`と、replay mean / K-medoidsまでfloat64を維持するdtype契約を修正。
- 修正版shard 0を同じcanonical kernelへversion 2としてpushし、Kaggle側bootstrap反映と`RUNNING`を確認。
- shard 1 canonicalの`Notebook not found`を記録し、限定recovery slugへversion 1をpush。Kaggle側bootstrap反映と`RUNNING`を確認。
- shard 2 canonicalの`Notebook not found`を記録し、限定recovery slugへversion 1をpush。Kaggle側bootstrap反映と`RUNNING`を確認。
- shard 3 canonicalの`Notebook not found`を記録し、限定recovery slugへversion 1をpush。Kaggle側bootstrap反映と`RUNNING`を確認。
- 修正版Kaggle CPU shard 0〜3の`COMPLETE`と全773 wells `ok`を確認。
- 3,783,989 ID / 773 wellsのunionと重複0を確認。
- 4 shardのreplay parityを集計し、difference RMSE 0.743077、最大差9.847657、exact parity falseを確認。
- schema SHA一致、validation source SHA不一致を確認し、strict mergeを棄却。
- reference集計でdirect medoid、oracle、cluster、hidden-like、worst-well guardを監査し、完了・不採用を決定。
- `metrics.json`、`result.md`、`SESSION_NOTES.md`、`README.md`、backlog、`experiment_summary.md`を更新。
- v3でPF入力のfloat32経由を除去し、canonical exp072 cache/schema SHA guardを追加。
- canonical exp072 inputとSHA固定exp209 parity controlを分離し、別名復元で`likpf_mean` merge衝突を解消。
- full 4 shard packageと`fba7683c` 1 well parity probe packageを再生成し、bootstrap契約を確認。
- ユーザー承認後にparity probe version 1をpushし、Kaggle側bootstrapと`RUNNING`を確認。
- v1のmissing `likpf_mean`前提によるPF前ERRORを修正し、同じkernelへversion 2をpush。
- parity probe v2を完走し、407/407 rowsでsaved exp072 `likpf_mean`とのexact parityを確認。
- 過去4 shardのCPU合計約9時間11分を提示し、ユーザーからfull 773 wellsを1 CPU notebookで実行する承認を得た。
- v3 full single-notebook version 1を完走し、3,783,989 rows / 773 wells、全well `ok`を確認。
- full replayがsaved exp072 `likpf_mean`と全3,783,989行でexact parityになることを確認。
- direct medoid、distance bucket、hidden-like、worst-well、row/block/whole-well oracle、cluster diagnosticsを監査。
- candidate generation仮説を支持、direct replacementを不採用としてexp243を完了。
- K8がall-K oracle headroomのほぼ全てを保持するため、後続候補をK8に限定。
