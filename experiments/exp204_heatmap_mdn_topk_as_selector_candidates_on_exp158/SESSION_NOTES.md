# exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158 セッションノート

## 2026-07-05 実装

目的: `heatmap_mdn_topk_as_selector_candidates_on_exp158` backlog を実装する。exp202 の heatmap MDN top10 path を、既存 selector の feature だけでなく selectable candidate として追加し、exp158/184 と同じ Viterbi continuity constraint 付きで選べるかを train-side audit する。

実装方針:

- route: `pf_beam`
- 親: `exp203_heatmap_mdn_candidates_into_selector_features`
- 比較基準: `exp184_cnn_sdf_mtp_heatmap_path_features_on_exp158`
- 追加入力: `exp202_heatmap_mdn_candidate_generator_probe_heatmap_candidates.csv.gz`
- selector 候補集合: 既存 8 候補 + `hmdn_top1` ... `hmdn_top10` = 18 候補
- LightGBM: 3 configs x 5 folds = 15 boosters
- parent/control retraining: なし
- inference / submit: なし

主な変更:

- `experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/` を exp203 から作成。
- `hmdn_top1_tvt` ... `hmdn_top10_tvt` を `ranker.candidates` に追加。
- 初期 exp099 cache には存在しない後段生成候補を `ranker.generated_candidate_columns` として扱い、required column check から除外。
- `add_heatmap_mdn_candidate_features` の candidate-distance feature は、候補追加後も既存 8 候補との距離として解釈できるよう、hmdn 候補を除いた base candidate set で計算。
- candidate-long feature に `candidate_is_hmdn_family`、`candidate_hmdn_rank`、`hmdn_candidate_rank01`、`hmdn_candidate_score`、`hmdn_confidence_x_candidate_hmdn_family` を追加。
- Viterbi `allowed_switch_candidates` に `hmdn_top1` ... `hmdn_top10` を追加。

leakage guard:

- `true_center_tvt`、`target_in_grid`、`best_mode`、abs-error、within10、oracle 系列は feature として読み込まない。
- hmdn topK は selector candidate としてのみ追加し、direct replacement、softmax weighted TVT、PF weight replacement、postprocess、submit には使わない。
- candidate-long label の absolute error は training target だけに使い、feature には入れない。

実行前ガード:

- active selector variant: 1 (`heatmap_mdn_topk_as_selector_candidates_on_exp158`)
- selectable candidates: 18
- selector heads: 3 (`lgb_multiclass`, `lgb_candidate_binary`, `lgb_candidate_error_ranker`)
- folds: 5
- planned boosters: 15
- Kaggle GPU: disabled (`runtime.kaggle.enable_gpu=false`)
- parent/control retraining: なし
- inference / submit: なし

未実行:

- Kaggle train push
- Kaggle output 取得

次の確認:

- py_compile / ruff F821 / Jupytext conversion / validate-exp
- Kaggle push 前に 18 candidates、15 boosters、control retraining なしを再確認する。

## 2026-07-05 静的検証

検証:

```bash
.venv/bin/python -m py_compile experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/heatmap_mdn_topk_as_selector_candidates_on_exp158.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_inference.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/settings.py
.venv/bin/ruff check experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/heatmap_mdn_topk_as_selector_candidates_on_exp158.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_inference.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/settings.py --select F821
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_inference.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_inference.py
make validate-exp EXP=exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158
```

結果:

- `py_compile`: pass
- `ruff --select F821`: pass
- Jupytext train / inference convert and `--test`: pass
- `make validate-exp`: pass
- candidate contract smoke: 18 candidates、last 10 are `hmdn_top1` ... `hmdn_top10`、initial required columns に `hmdn_top*_tvt` なし
- full feature smoke / Kaggle train: 未実行。exp099 / exp072 full cache は Kaggle source 前提で確認する。

注意:

- `py_compile` が生成した `__pycache__` は削除済み。
- Kaggle train / output 取得 / result 更新は未実行。

## 2026-07-06 Kaggle train 実行計画

ユーザー依頼により Kaggle Notebook train を実行する。推論 notebook、提出、Kaggle output archive 取得はこの段階では対象外。

実行前ガード:

- active selector variant: 1 (`heatmap_mdn_topk_as_selector_candidates_on_exp158`)
- selectable candidates: 18
- selector heads: 3 (`lgb_multiclass`, `lgb_candidate_binary`, `lgb_candidate_error_ranker`)
- folds: 5
- planned boosters: 15
- Kaggle GPU: disabled (`runtime.kaggle.enable_gpu=false`)
- parent/control retraining: なし
- inference / submit: なし

予定コマンド:

```bash
make validate-exp EXP=exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158
make prepare-kaggle-notebooks EXP=exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp204-heatmap-mdn-topk-as-selector-candidates-on-exp158-train --title 'exp204 heatmap mdn topk as selector candidates on exp158 train' --run-on-push --strict"
make push-kaggle-train EXP=exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158
```

実行結果:

- `make validate-exp`: pass
- 長い canonical kernel id `kentookumura/exp204-heatmap-mdn-topk-as-selector-candidates-on-exp158-train` は `SaveKernel` 400 で失敗。slug 長 62 文字で、exp200 の既知失敗例と同じ Kaggle 側 slug 制約の可能性が高い。
- 同じ exp のまま短縮 kernel id/title に再 prepare。
  - kernel id: `kentookumura/exp204-hmdn-topk-selector-train`
  - title: `exp204 hmdn topk selector train`
- `make push-kaggle-train`: `Kernel version 1 successfully pushed`
- URL: <https://www.kaggle.com/code/kentookumura/exp204-hmdn-topk-selector-train>
- `kaggle kernels pull kentookumura/exp204-hmdn-topk-selector-train -p /tmp/kaggle-pull/exp204-hmdn-topk-selector-train -m`: success
- initial `kaggle kernels logs`: Kaggle CLI version warning のみで本文空。実行中 logs 空は既知挙動のため失敗扱いしない。
- initial `kaggle kernels status`: `KernelWorkerStatus.RUNNING`
- 10 分監視用に `timeout 600 kaggle kernels logs -f --interval 30 kentookumura/exp204-hmdn-topk-selector-train` を開始したが、ユーザー指示によりローカル監視だけ停止。Kaggle notebook 実行自体は継続中。

## 2026-07-06 Kaggle train v1 失敗と修正

確認コマンド:

```bash
kaggle kernels status kentookumura/exp204-hmdn-topk-selector-train
kaggle kernels logs kentookumura/exp204-hmdn-topk-selector-train
```

結果:

- status: `KernelWorkerStatus.ERROR`
- logs: notebook setup と入力確認は完了し、`[fold 0] train=3026251 valid=757738` の直後に `nbclient.exceptions.DeadKernelError: Kernel died`。
- Python 例外ではなく kernel が落ちているため、Kaggle CPU runtime のメモリ不足と判断。

原因推定:

- exp204 は candidate を 8 から 18 に増やしたため、row-level の全 candidate pairwise 差分が 28 列から 153 列に増えた。
- `ranker.long_models.max_train_rows_per_fold=120000` は既に設定済みだったため、long model ではなく fold0 の row-wise multiclass 学習/特徴行列作成でメモリピークを踏んだ可能性が高い。

修正:

- `ranker.candidate_pairwise_scope: base_only` を追加し、row-level pairwise 差分は既存 base 候補間だけに制限。hmdn topK 関連は既存の `hmdn_*` score/rank/distance features と candidate-long features で扱う。
- `ranker.multiclass_lgbm.max_train_rows_per_fold: 500000`、`max_valid_rows_per_fold: 160000` を追加。multiclass は fold 内の sampled train/eval で学習し、OOF prediction は valid fold 全体へ実行する。
- `add_candidate_labels_and_features` を dict + `pd.concat` 方式に変更し、DataFrame fragmentation と不要なメモリ増加を抑制。

修正後チェック:

```bash
.venv/bin/python -m py_compile experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/heatmap_mdn_topk_as_selector_candidates_on_exp158.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_inference.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/settings.py
.venv/bin/ruff check experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/heatmap_mdn_topk_as_selector_candidates_on_exp158.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_inference.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/settings.py --select F821,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py
```

- `py_compile`: pass
- `ruff --select F821,E501`: pass
- Jupytext train convert and `--test`: pass

## 2026-07-06 Kaggle train v2

修正版を同じ kernel id に version 2 として push。

```bash
make validate-exp EXP=exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158
make prepare-kaggle-notebooks EXP=exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp204-hmdn-topk-selector-train --title 'exp204 hmdn topk selector train' --run-on-push --strict"
make push-kaggle-train EXP=exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158
kaggle kernels status kentookumura/exp204-hmdn-topk-selector-train
kaggle kernels logs kentookumura/exp204-hmdn-topk-selector-train
```

結果:

- `make validate-exp`: pass
- generated Kaggle package config includes `candidate_pairwise_scope: base_only`、multiclass sample cap `500000/160000`、long model sample cap `120000/120000`。
- `make push-kaggle-train`: `Kernel version 2 successfully pushed`
- URL: <https://www.kaggle.com/code/kentookumura/exp204-hmdn-topk-selector-train>
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: Kaggle CLI version warning のみで本文空。長時間監視はしない。

## 2026-07-07 exp212 full-grid backlog への defer

ユーザー確認と exp083 overlay 診断により、exp210 の covered-row artifact は exp072/exp083 の全 `md_since` 区間を覆う full-grid trajectory ではなく、selector が各行で通常候補として選ぶ入力には不足することが分かった。

判断:

- 旧 exp204 実装は exp202 row-interpolated heatmap paths を selectable candidate にする設計なので、このまま再実行しない。
- 当時の status は `deferred_pending_exp212_full_grid_artifact` とした。exp212 完了後に `deferred_ready_for_guarded_exp212_artifact` へ更新したが、後続判断で最終 status は `closed_rejected_heatmap_path_generation_route` になった。
- `KAGGLE_DIRECTION.md` に `exp212_heatmap_mdn_full_grid_path_generation_probe` を追加し、当時は exp204 系 selector candidate route を exp212 artifact の full-row coverage / continuity / oracle headroom が成立した後に再設計する想定だった。

## 2026-07-07 Kaggle train v2 timeout/ERROR と GPU 化

ユーザー報告: v2 が timeout。CLI logs を確認。

```bash
kaggle kernels status kentookumura/exp204-hmdn-topk-selector-train
kaggle kernels logs kentookumura/exp204-hmdn-topk-selector-train
```

結果:

- status: `KernelWorkerStatus.ERROR`
- v2 は `GPU enabled: False` の CPU runtime。
- fold0 では multiclass sampling が効き、`[fold 0] multiclass sampled train=500000/3026251 eval=160000/757738` まで進んだ。
- 5 folds と Viterbi grid 180 variants の最後まで進み、`[viterbi] evaluated 179/180 variants` まで出力。
- 最後の feature summary 保存時に `TypeError: arg must be a list, tuple, 1-d array, or Series` で失敗。

原因:

- `hmdn_top1_minus_last` など、heatmap MDN source feature と candidate delta feature の列名が衝突して DataFrame に duplicate column が発生していた。
- `summarize_feature_frame(frame, heatmap_mdn_columns)` で duplicate column access が 2D DataFrame になり、`pd.to_numeric` が TypeError。
- 実行時間も長く、CPU runtime のままでは 15 boosters + Viterbi 180 variants が実用的でない。

修正:

- Kaggle runtime を GPU/T4 に変更。
  - `runtime.kaggle.enable_gpu: true`
  - `runtime.kaggle.machine_shape: NvidiaTeslaT4`
- LightGBM 3 configs に GPU params を追加。
  - `device_type: gpu`
  - `gpu_use_dp: false`
  - `max_bin: 63`
- hmdn candidate delta feature は `candidate_hmdn_top*_minus_last` へ分離し、source feature と衝突しないようにした。
- `assert_unique_columns` を追加し、duplicate column は学習前に早期検出する。

v3 実行前ガード:

- active selector variant: 1 (`heatmap_mdn_topk_as_selector_candidates_on_exp158`)
- selectable candidates: 18
- selector heads: 3 (`lgb_multiclass`, `lgb_candidate_binary`, `lgb_candidate_error_ranker`)
- folds: 5
- planned boosters: 15
- Kaggle GPU: enabled (`NvidiaTeslaT4`)
- parent/control retraining: なし
- inference / submit: なし

v3 実行:

```bash
.venv/bin/python -m py_compile experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/heatmap_mdn_topk_as_selector_candidates_on_exp158.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_inference.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/settings.py
.venv/bin/ruff check experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/heatmap_mdn_topk_as_selector_candidates_on_exp158.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_inference.py experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/settings.py --select F821,E501
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158_train.py
make validate-exp EXP=exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158
make prepare-kaggle-notebooks EXP=exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158 EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp204-hmdn-topk-selector-train --title 'exp204 hmdn topk selector train' --run-on-push --strict"
kaggle kernels push -p experiments/exp204_heatmap_mdn_topk_as_selector_candidates_on_exp158/kaggle/train --accelerator NvidiaTeslaT4
kaggle kernels pull kentookumura/exp204-hmdn-topk-selector-train -p /tmp/kaggle-pull/exp204-hmdn-topk-selector-train-v3 -m
kaggle kernels status kentookumura/exp204-hmdn-topk-selector-train
kaggle kernels logs kentookumura/exp204-hmdn-topk-selector-train
```

結果:

- `py_compile`: pass
- `ruff --select F821,E501`: pass
- Jupytext train convert and `--test`: pass
- `make validate-exp`: pass
- generated Kaggle metadata: `enable_gpu=true`、`machine_shape=NvidiaTeslaT4`
- generated Kaggle config: LightGBM `device_type=gpu` / `max_bin=63` / duplicate-column guard included
- push: `Kernel version 3 successfully pushed`
- URL: <https://www.kaggle.com/code/kentookumura/exp204-hmdn-topk-selector-train>
- pulled Kaggle metadata: `enable_gpu=true`、`machine_shape=NvidiaTeslaT4`
- initial status: `KernelWorkerStatus.RUNNING`
- initial logs: Kaggle CLI version warning のみで本文空。長時間監視はしない。

## 2026-07-07 Kaggle train v3 中断

ユーザー判断により、実行中の v3 notebook は Kaggle 側で中断した。

理由:

- 後続 exp210 / exp212 の結果から、selector の通常候補として使うには、旧 exp204 の exp202 row-interpolated paths ではなく full-grid path artifact を正しく生成してから本実験を進めるべきと判断。
- exp212 full-grid artifact は row coverage 1.0 の contract を満たしたが fallback-heavy なので、単純に exp204 旧実装へ差し替えるのではなく coverage/fallback guard 付きで再設計する。

当時の扱い:

- v3 の途中結果は CV として採用しない。
- exp204 status は `deferred_ready_for_guarded_exp212_artifact`。
- 次に進める場合は、exp212 full-grid paths、`coverage_flag`、`fallback_flag`、`fill_method`、`candidate_score`、`source_window_count`、`overlap_weight`、`assignment_gap_flag` などを selector candidate / row-level confidence feature として取り込む設計に差し替える。
- 旧 exp202 row-interpolated candidate 実装はこのまま再実行しない。

## 2026-07-07 exp215 MTP full-tail artifact 待ちへ再分類

exp212 完了後の plot audit で、exp212 artifact は full-grid row coverage contract は満たすが、親 exp208 source が `max_tail_rows=2048` までで、後半は endpoint hold の直線 tail になることが分かった。

当時の扱い:

- exp204 は exp215 MTP full-tail artifact 待ちに更新した。
- exp204 は exp212 heuristic full-grid artifact を通常 selector candidate として使わない。
- 次に進める場合は、`exp215_mtp_full_tail_heatmap_path_generator_probe` の learned MTP full-tail artifact を入力にする。
- exp215 artifact には `path_logit`、`path_prob`、`weighted_tvt_pred`、source/fallback flags を含める想定だった。
- exp215 が fallback-heavy、weighted path が単体で弱い、または hidden-like / worst-well / near-row を壊す場合、exp204 candidate route は再開せず feature-only に戻す。

## 2026-07-07 exp215 完了後の再分類

exp215 Kaggle train v1 が完了した。

- full-grid source coverage: 1.0
- fallback unique row rate: 0.0
- existing union oracle RMSE: 7.434029932
- existing + learned MTP top5 oracle RMSE: 5.113654814
- learned MTP top5 only oracle RMSE: 32.333142886
- learned MTP weighted oracle RMSE: 59.272141581

当時の扱い:

- exp204 status は `deferred_ready_for_guarded_exp215_mtp_topk_artifact` に更新した。
- exp215 topK path artifact は selector candidate として再設計候補にした。
- exp215 weighted path は単体で弱いため、通常 selectable candidate にはしない。使う場合は confidence / disagreement / audit feature に限定する。
- 旧 exp202 row-interpolated candidate 実装や exp212 fallback-heavy artifact では再実行しない。

## 2026-07-07 heatmap path generation route を close

ユーザー判断により、heatmap から path を生成するアイデア自体を閉じる。

理由:

- exp202/207/208/210/212/215 で oracle headroom は見えたが、生成 path 単体は弱く、selector の通常候補として採用する根拠が不足した。
- exp212 は full-grid contract を満たしたが fallback-heavy / endpoint hold tail が残った。
- exp215 は fallback unique row rate 0.0 を達成したが、learned MTP top5 only RMSE 32.333142886、weighted path RMSE 59.272141581 と弱かった。
- exp203 feature-only も exp184 best を更新しなかった。

現在の扱い:

- exp204 status は `closed_rejected_heatmap_path_generation_route` に更新する。
- `heatmap_mdn_topk_as_selector_candidates_on_exp158` は active backlog から外す。
- exp204 は再実行しない。
- exp202 row-interpolated、exp212 heuristic full-grid、exp215 learned MTP full-tail のいずれも selector candidate、direct replacement、softmax weighted TVT、PF weight replacement、postprocess、inference port、submit には使わない。
- exp202/203/207/208/210/212/215 artifacts は diagnostic history として保持する。
