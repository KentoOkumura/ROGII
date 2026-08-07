# exp197_cnn_pf_likelihood_probe セッションノート

## 目的

`cnn_pf_likelihood_probe` backlog の実装。discussion 699853 の「PF point-GR likelihood を learned local CNN/SDF likelihood に置き換える」案を、live PF weight replacement ではなく、exp099 fixed candidates 上の train-side frozen candidate scorer として検証する。

## 現在の状態

- Route: pf_beam
- 状態: Kaggle train version 1 完了
- CV: candidate AUC real_gr learned_prob 0.9086916392
- LB: まだなし

## コマンドログ

実行したコマンドを時系列で記録します。未実行のコマンドは予定として明記します。

### 実行済み

```bash
task new-steering EXP=exp197_cnn_pf_likelihood_probe
task new-exp EXP=exp197_cnn_pf_likelihood_probe
make new-steering EXP=exp197_cnn_pf_likelihood_probe
make new-exp EXP=exp197_cnn_pf_likelihood_probe
.venv/bin/python -m py_compile experiments/exp197_cnn_pf_likelihood_probe/exp197_cnn_pf_likelihood_probe_train.py
.venv/bin/python -m py_compile experiments/exp197_cnn_pf_likelihood_probe/exp197_cnn_pf_likelihood_probe_inference.py
.venv/bin/ruff check experiments/exp197_cnn_pf_likelihood_probe/exp197_cnn_pf_likelihood_probe_train.py experiments/exp197_cnn_pf_likelihood_probe/exp197_cnn_pf_likelihood_probe_inference.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp197_cnn_pf_likelihood_probe/exp197_cnn_pf_likelihood_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp197_cnn_pf_likelihood_probe/exp197_cnn_pf_likelihood_probe_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp197_cnn_pf_likelihood_probe/exp197_cnn_pf_likelihood_probe_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp197_cnn_pf_likelihood_probe/exp197_cnn_pf_likelihood_probe_inference.py
make validate-exp EXP=exp197_cnn_pf_likelihood_probe
make prepare-kaggle-notebooks EXP=exp197_cnn_pf_likelihood_probe EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp197-cnn-pf-likelihood-probe-train --title 'exp197 cnn pf likelihood probe train' --run-on-push --strict"
make push-kaggle-train EXP=exp197_cnn_pf_likelihood_probe
kaggle kernels logs kentookumura/exp197-cnn-pf-likelihood-probe-train
timeout 300 kaggle kernels logs -f --interval 20 kentookumura/exp197-cnn-pf-likelihood-probe-train
kaggle kernels status kentookumura/exp197-cnn-pf-likelihood-probe-train
timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp197-cnn-pf-likelihood-probe-train
kaggle kernels logs kentookumura/exp197-cnn-pf-likelihood-probe-train
kaggle kernels status kentookumura/exp197-cnn-pf-likelihood-probe-train
kaggle kernels logs kentookumura/exp197-cnn-pf-likelihood-probe-train
kaggle kernels status kentookumura/exp197-cnn-pf-likelihood-probe-train
kaggle kernels output kentookumura/exp197-cnn-pf-likelihood-probe-train -p /tmp/exp197_kaggle_output_v1
```

`task` は未導入で失敗したため、`make` fallback で scaffold を作成した。`validate-exp` と strict prepare は通過済み。

2026-07-05 09:33 JST 時点で Kaggle train は version 1 push 成功。実行 URL は <https://www.kaggle.com/code/kentookumura/exp197-cnn-pf-likelihood-probe-train>。`kaggle kernels status` は `KernelWorkerStatus.RUNNING`。`kaggle kernels logs` と 5 分 / 10 分 follow は warning 以外の notebook 出力なし。

2026-07-05 にユーザーの完了連絡後、`kaggle kernels status` で `KernelWorkerStatus.COMPLETE` を確認。logs から decision / SHA を確認し、topK と full metrics を読むために output を `/tmp/exp197_kaggle_output_v1` へ取得した。output 取得理由は、logs に `display(topk_df)` の表が十分出ず、topK / summary / artifact SHA の実ファイル確認が必要だったため。

## 変更点

- `config.yaml` を PF/Beam route の train-side GPU diagnostic として設定。
- `exp197_cnn_pf_likelihood_probe_train.py` を Jupytext percent notebook として追加。入力確認、candidate index、CNN training、negative control、metrics / SHA 保存までを notebook 上で追える構成。
- `exp197_cnn_pf_likelihood_probe_inference.py` は推論なし guard のみ。
- 比較 baseline は `likpf_mean_single`、point-GR likelihood、exp099 multiobs score、exp111 learned likelihood。
- 直接 PF weight replacement、raw-test feature generation、submit は対象外。

## 再現性メモ

- seed policy: fixed global seed + SHA256 stable row subsample / shuffled-GR roll。
- stochastic components: upstream exp099 PF/Beam cache、PyTorch CUDA conv、AdamW、DataLoader shuffle。
- CPU/GPU runtime: Kaggle T4 GPU 前提。CPU fallback disabled。
- Kaggle kernel id / version: `kentookumura/exp197-cnn-pf-likelihood-probe-train` version 1 COMPLETE。
- input / feature schema SHA: exp099 source decompressed `1939d536b1e56f7c0ea3847cc386ef769b0d33759d16e816c9ce180f0532df9a`, exp099 schema `203e4f9a280fe901f5f21d39b85c3e0e2a7fe10c466081c15015c7fb014a0413`, exp111 decompressed `3aa5e72e982417012a18f4172df1a233ef0f609cf91d48fb1250fc74fa9e89f8`。
- feature content SHA: candidate index gz `e48532343a61209be23e6028540e15ae3c19bc1751d3b2ff169311441fcb32fd`, decompressed `d78918f24934c394c40e883cad2207e1c66b629f9ec1bb3146cda8f95fae4e6b`。
- model manifest / model SHA: manifest `8e2b0b3b97954b1bd95e6514e4ae750702fa1d7f46ad44efd34b3e07b4ed1188`。variant `.pt` SHA は manifest 内に記録。
- prediction SHA: OOF probability `29795eeb9d5771097b611ad2a66a19ee58c172bf05a3b323dc6862bdefb88e59`, expected error `cf5a2a6820498acd768a1267dbc82080ec35ea9a7c673d161871246001f83634`。
- submission SHA: 対象外。submission は作らない。
- rerun check: 未実施。deterministic anchor とは扱わない。

## Kaggle train v1 result

- status: COMPLETE
- metric: candidate AUC
- real_gr learned_prob AUC: 0.9086916392
- shuffled_gr learned_prob AUC: 0.9027273274
- no_gr learned_prob AUC: 0.9053030435
- real - shuffled AUC: +0.0059643118
- real - no_gr AUC: +0.0033885957
- exp111 learned probability AUC: 0.9158250218
- exp099 multiobs score AUC: 0.6121555671
- point-GR likelihood AUC: 0.5690632201
- real_gr learned_prob top1 RMSE / MAE / within10: 11.301053 / 6.735788 / 0.784917
- likPF single top1 RMSE / MAE / within10: 11.293248 / 6.764031 / 0.785750
- learned_error top1 RMSE / MAE / within10: 11.252965 / 6.708288 / 0.784500
- real_gr learned_prob top2 / top3 / top5 oracle RMSE: 9.420265 / 8.117179 / 7.774709
- notebook decision: `weak_real_gr_signal_needs_guarded_followup`

Interpretation: CNN scorer は候補識別 AUC としては強いが、real GR の上積みは shuffled/no-GR に対して小さい。top1 は likPF single と同等以下で、exp111 learned probability AUC も下回るため、PF weight replacement や submit へは進めない。

## Kaggle train cost guard

- active variants: 3 (`real_gr`, `shuffled_gr`, `no_gr`)
- folds: 1
- model/config 数: 3 PyTorch CNN models
- LightGBM config 数: 0
- 合計 booster 数: 0
- 親実験 control / baseline の再学習: なし
- PF/Beam 再生成: なし。exp099 fixed candidate cache を読むだけ。
- GPU runtime: Kaggle T4、epochs 3、train row subsample 40,000 rows x 5 candidates、valid row subsample 12,000 rows x 5 candidates。

## 次のアクション

1. exp197 は train-side diagnostic として完了。
2. 追加するなら、candidate scalar / row context を制限した GR-only ablation で real GR の純粋寄与を確認する。ただし現時点では高優先にしない。
