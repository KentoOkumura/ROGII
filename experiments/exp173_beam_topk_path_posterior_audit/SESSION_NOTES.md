# exp173_beam_topk_path_posterior_audit セッションノート

## 目的

`beam_topk_path_posterior_audit` バックログを実装し、Beam search 本体が保持した top-K path と path cost から posterior 候補と診断指標を保存できる状態にする。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_negative_no_submit
- CV: best posterior RMSE 15.972927962
- LB: なし
- Kaggle train: v2 完了
- inference / submit: 対象外

## Push 前の計算規模

- active Beam variants: 3
- LightGBM config 数: 0
- fold 数: 0
- 合計 booster 数: 0
- control / 親実験再学習: なし
- GPU: 不使用

## コマンドログ

```bash
make new-steering EXP=exp173_beam_topk_path_posterior_audit
make new-exp EXP=exp173_beam_topk_path_posterior_audit
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp173_beam_topk_path_posterior_audit/exp173_beam_topk_path_posterior_audit_train.py
make validate-exp EXP=exp173_beam_topk_path_posterior_audit
make prepare-kaggle-notebooks EXP=exp173_beam_topk_path_posterior_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp173-beam-topk-path-posterior-audit-train --title 'exp173 beam topk path posterior audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp173_beam_topk_path_posterior_audit
kaggle kernels pull kentookumura/exp173-beam-topk-path-posterior-audit-train -p /tmp/kaggle-pull/exp173-beam-topk-path-posterior-audit-train -m
kaggle kernels logs kentookumura/exp173-beam-topk-path-posterior-audit-train
kaggle kernels status kentookumura/exp173-beam-topk-path-posterior-audit-train
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp173_beam_topk_path_posterior_audit/exp173_beam_topk_path_posterior_audit_train.py
make prepare-kaggle-notebooks EXP=exp173_beam_topk_path_posterior_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp173-beam-topk-path-posterior-audit-train --title 'exp173 beam topk path posterior audit train' --run-on-push --strict"
make push-kaggle-train EXP=exp173_beam_topk_path_posterior_audit
```

`task` はこの環境に入っていなかったため、Makefile の同等ターゲットを使った。

2026-07-03:

- Kaggle train kernel: `kentookumura/exp173-beam-topk-path-posterior-audit-train`
- URL: https://www.kaggle.com/code/kentookumura/exp173-beam-topk-path-posterior-audit-train
- pushed version: 1
- metadata: CPU、internet off、GPU off、competition source `rogii-wellbore-geology-prediction`、kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train`
- post-push pull: 成功。`/tmp/kaggle-pull/exp173-beam-topk-path-posterior-audit-train`
- status: `KernelWorkerStatus.RUNNING`
- logs: 実行中のため CLI logs は空。空ログだけでは失敗判定しない。

2026-07-03 v1 failure / v2 rerun:

- v1 status: `KernelWorkerStatus.ERROR`
- v1 error: `ValueError: No kernel name found in notebook and no override provided.`
- cause: Jupytext 変換後の train notebook に `kernelspec` metadata がなかったため、Papermill 起動前に失敗した。
- fix: `exp173_beam_topk_path_posterior_audit_train.py` に `python3` kernelspec metadata を追加し、notebook と Kaggle package を再生成した。
- v2 pushed: same kernel id `kentookumura/exp173-beam-topk-path-posterior-audit-train`

2026-07-03 v2 result:

- v2 status: `KernelWorkerStatus.COMPLETE`
- rows / wells: 3,783,989 rows / 773 wells
- runtime: 約 1,568 sec
- input feature cache decompressed SHA: `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`
- primary baseline: `likpf_mean` RMSE 11.594897672
- best posterior: `beam_topk_sm11_bw64_posterior_mean_t16`
  - RMSE 15.972927962
  - MAE 10.852413073
  - within10 0.602145249
  - delta vs `likpf_mean` +4.378030290
  - max well regression vs `likpf_mean` +57.494505670
- best top-K oracle: `beam_topk_sm11_bw64_topk_oracle`
  - RMSE 15.549454381
  - MAE 10.305776697
  - within10 0.626111228
  - delta vs `likpf_mean` +3.954556709
  - max well regression vs `likpf_mean` +57.493741442
- generated SHA:
  - candidate metrics `87126c3131a861616957732f9e3c3a57a32c526418bef072300e5453928d7e99`
  - bucket metrics `1d12a13ed94b63a50c1e5fcaba18f5f54cb060e3f2769002dafc4b47329bbd59`
  - by-well `02bd2ff24cf26ee9becf8610c93832188404651fb5cdbf9264af9f659cce7334`
  - group metrics `09fc984303a4a153d69e46740cf0ddda510f06c6eb16cc1fdecd0d17f64b2dab`
  - beam quality `d17324f867b826b774b2f73cbfa07047a1537a16e51403ff2b59f2dcf29b9a64`
  - top-K diagnostics decompressed `08b1ed91742e4352b732fb739fdd59a8b4c53f53582f8a1c295a7b123e070301`
  - top-K paths decompressed `cf23b20a5b2ee9c8266f6272374463ec49cf8229c78570b73908cd346f4c73cc`
  - candidate wide decompressed `f993aaed3f59a39f3e367e1c18b3a7a394a254db09c1a5277d90d605621613bd`
- warning: `DataFrame is highly fragmented` が大量に出たが、実行は完了した。再実行する場合は column insert を dict/concat に直す余地がある。
- decision: negative。best posterior も top-K oracle も `likpf_mean` に大きく届かないため、inference port / submit はしない。

## 変更点

- `beam_topk_path_posterior_audit.py` を追加し、exp146 の入力読込・評価枠をもとに Beam top-K path/cost 復元へ変更した。
- `config.yaml` を pf_beam route の train-side audit として更新した。
- Jupytext percent 形式の train script を追加し、正規 train notebook を生成した。
- candidate metrics、bucket metrics、by-well metrics、group metrics、beam quality、top-K diagnostics、candidate wide、summary JSON を出力する設計にした。

## 再現性メモ

- seed policy: 新規乱数なし。Beam dynamic programming は deterministic。
- stochastic components: 上流 exp072 PF/Beam/likPF cache のみ。
- CPU/GPU runtime: CPU、GPU 不使用、LightGBM 学習なし。
- Kaggle kernel id / version: `kentookumura/exp173-beam-topk-path-posterior-audit-train` v2 complete。
- input / feature schema SHA: source decompressed SHA `99a3c70a19b2f22a8c76c5947b14692e7b8207ea45f38b8b1c5f327d320e1350`。
- feature content SHA: gzip 生成物は decompressed content SHA を主証拠にする。
- model manifest / model SHA: model なし。
- prediction SHA: prediction なし。
- submission SHA: submission なし。
- rerun check: train-side audit のため、採用候補になった場合だけ追加実施する。

## 次のアクション

1. `result.md`、`metrics.json`、`experiment_summary.md`、`backlog/KAGGLE_DIRECTION.md` に結果を反映する。
2. Beam top-K posterior backlog は完了/不採用として閉じる。
