# exp203_heatmap_mdn_candidates_into_selector_features セッションノート

## 2026-07-05 実装

目的: `heatmap_mdn_candidates_into_selector_or_ml_features` backlog のうち、selector 入力化を実装する。ユーザー確認により、heatmap MDN topK を新しい selectable candidate にはせず、既存 selector への add-only feature として扱う。

実装方針:

- route: `pf_beam`
- 親: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- 追加入力: `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidates.csv.gz`
- selector 候補集合: exp184 と同じ 8 候補で固定
- 追加 feature block: `hmdn_`
- LightGBM: 3 configs x 5 folds = 15 boosters
- parent/control retraining: なし
- inference / submit: なし

追加した feature:

- exp202 topK TVT / score / score mass / entropy / margin / spread
- well 内 `row_center` 線形補間による row-level 展開
- topK と既存候補集合の最小距離、within10 proxy
- candidate-long の `hmdn_*_minus_candidate` / abs / normalized abs
- hmdn confidence proxy と PF/Beam / dense family interaction
- hmdn sparse-distance / confidence bucket 診断

leakage guard:

- `true_center_tvt`、`target_in_grid`、`best_mode`、abs-error、within10、oracle 系列は feature として読み込まない。
- heatmap MDN topK は direct TVT replacement、weighted average、PF weight、postprocess、submit に使わない。
- candidate-long label の absolute error は training target だけに使い、feature には入れない。

実装メモ:

- `docs/legacy/steering/20260705-exp203-heatmap-mdn-candidates-into-selector-features/` を作成。
- `experiments/exp203_heatmap_mdn_candidates_into_selector_features/` を exp184 から作成。
- 実装本体を `heatmap_mdn_candidates_into_selector_features.py` にリネームし、exp202 source reader と `hmdn_` feature block を追加。

未実行:

- Kaggle train push
- Kaggle output 取得

次の確認:

- py_compile / ruff F821 / Jupytext conversion / validate-exp
- Kaggle push 前に active variant 1、15 boosters、control retraining なしを再確認する。

## 2026-07-05 静的検証

検証:

```bash
.venv/bin/python -m py_compile experiments/exp203_heatmap_mdn_candidates_into_selector_features/heatmap_mdn_candidates_into_selector_features.py experiments/exp203_heatmap_mdn_candidates_into_selector_features/exp203_heatmap_mdn_candidates_into_selector_features_train.py experiments/exp203_heatmap_mdn_candidates_into_selector_features/exp203_heatmap_mdn_candidates_into_selector_features_inference.py experiments/exp203_heatmap_mdn_candidates_into_selector_features/settings.py
.venv/bin/ruff check experiments/exp203_heatmap_mdn_candidates_into_selector_features/heatmap_mdn_candidates_into_selector_features.py experiments/exp203_heatmap_mdn_candidates_into_selector_features/exp203_heatmap_mdn_candidates_into_selector_features_train.py experiments/exp203_heatmap_mdn_candidates_into_selector_features/exp203_heatmap_mdn_candidates_into_selector_features_inference.py experiments/exp203_heatmap_mdn_candidates_into_selector_features/settings.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp203_heatmap_mdn_candidates_into_selector_features/exp203_heatmap_mdn_candidates_into_selector_features_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp203_heatmap_mdn_candidates_into_selector_features/exp203_heatmap_mdn_candidates_into_selector_features_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp203_heatmap_mdn_candidates_into_selector_features/exp203_heatmap_mdn_candidates_into_selector_features_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp203_heatmap_mdn_candidates_into_selector_features/exp203_heatmap_mdn_candidates_into_selector_features_inference.py
make validate-exp EXP=exp203_heatmap_mdn_candidates_into_selector_features
```

結果:

- `py_compile`: pass
- `ruff --select F821`: pass
- Jupytext train / inference convert and `--test`: pass
- `make validate-exp`: pass
- exp202 source reader smoke: 10,822 rows / 773 wells / 52 `hmdn_` source features、decompressed SHA `3da094b0530ce1be289617e0d00ec6c667e608a36cf04d44c6a353ab809a2dba`

注意:

- local full feature smoke は exp072 full replay cache が workspace にないため実施していない。Kaggle train では `kentookumura/exp072-exp063-full-replay-feature-cache-train` を source に含める。
- Kaggle train / output 取得 / result 更新は未実行。

## 2026-07-05 Kaggle train push

ユーザー指示により Kaggle train を実行する。

実行前ガード:

- active selector variant: 1 (`heatmap_mdn_candidates_into_selector_features`)
- selector heads: 3 (`lgb_multiclass`, `lgb_candidate_binary`, `lgb_candidate_error_ranker`)
- folds: 5
- planned boosters: 15
- Kaggle GPU: disabled (`runtime.kaggle.enable_gpu=false`)
- parent/control retraining: なし
- inference / submit: なし

予定 kernel:

- id: `kentookumura/exp203-hmdn-selector-train`
- title: `exp203 hmdn selector train`

実行:

```bash
make validate-exp EXP=exp203_heatmap_mdn_candidates_into_selector_features
make prepare-kaggle-notebooks EXP=exp203_heatmap_mdn_candidates_into_selector_features EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp203-hmdn-selector-train --title 'exp203 hmdn selector train' --run-on-push --strict"
make push-kaggle-train EXP=exp203_heatmap_mdn_candidates_into_selector_features
kaggle kernels status kentookumura/exp203-hmdn-selector-train
kaggle kernels logs kentookumura/exp203-hmdn-selector-train
```

結果:

- `make validate-exp`: pass
- `prepare-kaggle-notebooks`: pass
- `push-kaggle-train`: Kaggle kernel version 1 pushed
- URL: https://www.kaggle.com/code/kentookumura/exp203-hmdn-selector-train
- 初回 status: `KernelWorkerStatus.RUNNING`
- 初回 logs: Kaggle CLI warning のみで notebook logs はまだ未出力
- 2 分後の再確認 status: `KernelWorkerStatus.RUNNING`
- 2 分後の logs: Kaggle CLI warning のみで notebook logs はまだ未出力
- `timeout 300 kaggle kernels logs -f --interval 20 ...` で短時間監視を開始したが、ユーザー指示により手元の監視プロセスを停止した。

## 2026-07-06 Kaggle train 完了確認

ユーザー連絡により exp203 train 完了を確認した。

確認:

```bash
kaggle kernels status kentookumura/exp203-hmdn-selector-train
kaggle kernels logs kentookumura/exp203-hmdn-selector-train
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- kernel: `kentookumura/exp203-hmdn-selector-train` version 1
- URL: https://www.kaggle.com/code/kentookumura/exp203-hmdn-selector-train
- rows / wells: 3,783,989 / 773
- runtime: 36,287.401844 sec
- feature count: 298
- `hmdn_` generated feature count: 75
- exp202 heatmap MDN source rows / wells: 10,822 / 773
- exp202 heatmap MDN source decompressed SHA: `3da094b0530ce1be289617e0d00ec6c667e608a36cf04d44c6a353ab809a2dba`
- source valid rate: 1.0
- nearest sample distance p95: 3,522 rows

best Viterbi:

- variant: `viterbi_sw050_bias000_jw100_jf025_d0150_std999999_md0000_seg001`
- RMSE: 10.665741318
- MAE: 6.350286735
- within10: 0.797977743
- oracle label accuracy: 0.271556551
- path switches: 12,807
- path switches / 1000 rows: 3.384523581
- max path switches / 1000 rows: 22.604191194
- default candidate rate: 0.433944708
- pf_ancc selection rate: 0.368075859

比較:

- vs `likpf_mean`: -0.929156354 RMSE
- vs exp158 continuity: -0.123421935 RMSE
- vs exp184 best Viterbi: +0.105090994 RMSE
- exp184 path switches は 5,713 / 1.509782 per 1000 rows だったため、exp203 は RMSE と path switch の両方で exp184 を更新しない。

生成物 SHA:

- metrics: `cd59937eb3d6ef7c2943009d5f42450b1587d796164b9592bf5f29bf4a4d8c69`
- predictions: `751ee27338d36cbbd6a485d413ab6b891c8884322a419be01a19a85c8b231f5f`
- predictions decompressed: `65d1b1c9120f61e8200a723560608344036bd8d4e1e0d426efe5728b505d1cc5`
- feature schema: `104946426942036c2a6f77bf2be2266acdd467f1121d46d353821b5bbb7d32fc`
- heatmap MDN feature schema: `471ddda01b71d16150efd7fbf1fc84f32d4a9d855be91c5d64622093f1684543`
- model manifest: `dee02204811ec897564cfd5c3c0dd0192632b1d153a89c3fad21c4ac76b51f26`

判断:

- exp202 heatmap MDN add-only feature signal は exp158 からの改善として確認できた。
- ただし親の exp184 より悪いため、exp203 は inference / submit に進めない。
- ユーザー意図に近い heatmap MDN path の selectable candidate 化は `exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158` で扱う。
- Kaggle output archive は取得していない。CV と summary は Kaggle logs / notebook 出力の summary JSON を根拠に記録した。
