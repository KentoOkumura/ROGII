# exp131_gr_shape_descriptor_matching_ablation セッションノート

## 目的

`gr_shape_descriptor_matching_ablation` backlog を実装する。単点 GR / NCC / banded local shift / shape descriptor cost を、同じ exp072 PF/Beam 候補 surface 上で比較し、real GR が shuffled-GR / no-GR negative control を上回るかを train-side pseudo-tail で監査する。

## 現在の状態

- Route: pf_beam
- 状態: completed_train_side_audit
- CV: candidate likelihood ablation only
- LB: なし
- 提出: なし

## コマンドログ

### 2026-06-26 JST 実装

```bash
uv run python scripts/new_steering.py --experiment exp131_gr_shape_descriptor_matching_ablation
uv run python scripts/new_experiment.py --name exp131_gr_shape_descriptor_matching_ablation --source experiments/exp099_pf_multi_observation_likelihood_probe
mv experiments/exp131_gr_shape_descriptor_matching_ablation/exp099_pf_multi_observation_likelihood_probe_train.ipynb experiments/exp131_gr_shape_descriptor_matching_ablation/exp131_gr_shape_descriptor_matching_ablation_train.ipynb
mv experiments/exp131_gr_shape_descriptor_matching_ablation/exp099_pf_multi_observation_likelihood_probe_inference.ipynb experiments/exp131_gr_shape_descriptor_matching_ablation/exp131_gr_shape_descriptor_matching_ablation_inference.ipynb
mv experiments/exp131_gr_shape_descriptor_matching_ablation/pf_multi_observation_likelihood_probe.py experiments/exp131_gr_shape_descriptor_matching_ablation/gr_shape_descriptor_matching_ablation.py
```

実装内容:

- `docs/legacy/steering/20260626-exp131-gr-shape-descriptor-matching-ablation/` を作成した。
- `config.yaml` を shape descriptor matching ablation 用に更新した。
- `gr_shape_descriptor_matching_ablation.py` を実装し、既存候補 `pf_ancc` / `beam_mean` / `likpf_mean` / `sc_ens` / `hyb` に対する descriptor score を生成するようにした。
- score variants は `raw_point_real`、`ncc_window_real`、`banded_shift_real`、`shape_descriptor_real`、`combo_descriptor_real`、`combo_descriptor_shuffled`、`no_gr_constant`。
- train notebook は設定確認、監査実行、出力 preview、metrics 保存のセル構成に更新する。
- inference notebook は診断専用 no-op として明記する。

## 検証予定

```bash
python3 -m py_compile experiments/exp131_gr_shape_descriptor_matching_ablation/gr_shape_descriptor_matching_ablation.py experiments/exp131_gr_shape_descriptor_matching_ablation/settings.py
python3 -m json.tool experiments/exp131_gr_shape_descriptor_matching_ablation/exp131_gr_shape_descriptor_matching_ablation_train.ipynb >/tmp/exp131_train_nb.json
uv run ruff check experiments/exp131_gr_shape_descriptor_matching_ablation/gr_shape_descriptor_matching_ablation.py experiments/exp131_gr_shape_descriptor_matching_ablation/settings.py
uv run python scripts/validate_experiment.py --experiment exp131_gr_shape_descriptor_matching_ablation
```

結果:

- `python3 -m py_compile`: PASS
- notebook JSON validation: PASS
- `uv run ruff check`: PASS
- `uv run python scripts/validate_experiment.py --experiment exp131_gr_shape_descriptor_matching_ablation`: PASS

補助スクリプト smoke:

```bash
PYTHONPATH=experiments/exp131_gr_shape_descriptor_matching_ablation uv run python experiments/exp131_gr_shape_descriptor_matching_ablation/gr_shape_descriptor_matching_ablation.py --max-rows 1000 --output-dir /tmp/exp131_gr_shape_descriptor_smoke
```

結果:

- 実行環境の `uv` 起動は PASS。
- ローカルに exp072 train cache がないため `FileNotFoundError` で停止。
- 参照予定 input は Kaggle kernel source `kentookumura/exp072-exp063-full-replay-feature-cache-train` にあるため、これはローカル smoke の入力不足として扱う。

Kaggle package:

```bash
uv run python scripts/prepare_kaggle_notebooks.py --experiment exp131_gr_shape_descriptor_matching_ablation --notebook train --kernel-id kentookumura/exp131-gr-shape-descriptor-train --title 'exp131 gr shape descriptor train' --run-on-push --strict
```

結果:

- package 生成: PASS
- 生成先: `experiments/exp131_gr_shape_descriptor_matching_ablation/kaggle/train`
- kernel id: `kentookumura/exp131-gr-shape-descriptor-train`
- title: `exp131 gr shape descriptor train`
- GPU: false
- internet: false
- kernel source: `kentookumura/exp072-exp063-full-replay-feature-cache-train`

## 生成予定ファイル

- `exp131_gr_shape_descriptor_matching_ablation_descriptor_scores_train_features.csv.gz`
- `exp131_gr_shape_descriptor_matching_ablation_descriptor_scores_feature_schema.csv`
- `exp131_gr_shape_descriptor_matching_ablation_candidate_metrics.csv`
- `exp131_gr_shape_descriptor_matching_ablation_score_variant_metrics.csv`
- `exp131_gr_shape_descriptor_matching_ablation_rank_metrics.csv`
- `exp131_gr_shape_descriptor_matching_ablation_bucket_metrics.csv`
- `exp131_gr_shape_descriptor_matching_ablation_by_well.csv`
- `exp131_gr_shape_descriptor_matching_ablation_descriptor_well_summary.csv`
- `exp131_gr_shape_descriptor_matching_ablation_row_context.csv.gz`
- `exp131_gr_shape_descriptor_matching_ablation_candidate_long.csv.gz`
- `exp131_gr_shape_descriptor_matching_ablation_summary.json`

## 注意

この実験は train-side diagnostic only。直接 TVT 候補、ML add-only 大量投入、inference port、submission は作らない。real GR が shuffled/no-GR controls を明確に上回らない場合は backlog を閉じる。

### 2026-06-26 JST Kaggle train v1 push

```bash
kaggle kernels push -p experiments/exp131_gr_shape_descriptor_matching_ablation/kaggle/train
kaggle kernels pull kentookumura/exp131-gr-shape-descriptor-train -p /tmp/kaggle-pull/exp131-gr-shape-descriptor-train -m
kaggle kernels logs kentookumura/exp131-gr-shape-descriptor-train
timeout 180 kaggle kernels logs -f --interval 15 kentookumura/exp131-gr-shape-descriptor-train
kaggle kernels status kentookumura/exp131-gr-shape-descriptor-train
kaggle kernels output kentookumura/exp131-gr-shape-descriptor-train -p experiments/exp131_gr_shape_descriptor_matching_ablation/kaggle/output/train_v1
```

結果:

- Kernel version 1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp131-gr-shape-descriptor-train
- `kaggle kernels pull ... -m` 成功。canonical id の存在確認済み。
- 初期 `logs` は warning 以外は空。
- `logs -f` は 180 秒間出力なしで timeout。過去実験同様、Kaggle API log が空で返っている可能性がある。
- `kaggle kernels status` は `KernelWorkerStatus.RUNNING`。
- `kaggle kernels output .../train_v1` は戻ったが、2026-06-26 20:21 JST 時点で output file はまだなし。

次:

- 完了後に通常 `logs` と `output` を同じ kernel id で再取得する。
- `summary.json`、score variant metrics、rank metrics、bucket / by-well stress を確認して `metrics.json` / `result.md` / `KAGGLE_DIRECTION.md` を更新する。

### 2026-06-26 JST Kaggle train v1 完了後 output 確認

ユーザー完了連絡後に logs / output を取得した。

```bash
kaggle kernels logs kentookumura/exp131-gr-shape-descriptor-train
kaggle kernels output kentookumura/exp131-gr-shape-descriptor-train -p experiments/exp131_gr_shape_descriptor_matching_ablation/kaggle/output/train_v1
kaggle kernels status kentookumura/exp131-gr-shape-descriptor-train
```

結果:

- status: `KernelWorkerStatus.COMPLETE`
- logs: 取得成功。notebook runtime は 5,987.899 sec。
- output: 大きい `descriptor_scores_train_features.csv.gz` の途中で長時間止まったため中断した。ローカル取得済みは candidate metrics、bucket metrics、by-well、candidate long、feature schema など一部。
- Kaggle logs 上では全 expected files が生成済みで、wide train feature cache は 1,790,873,153 bytes。
- logs stdout から summary を `/tmp/exp131_summary_from_logs.json` に抽出した。

主要結果:

- rows / wells: 3,783,989 / 773
- best single candidate: `likpf_mean` RMSE 11.594897 / MAE 7.067633 / within10 0.772807
- baseline oracle: RMSE 7.434030 / within10 0.906525
- best score variant: `combo_descriptor_real` AUC 0.659206 / logloss 0.653906
- negative controls:
  - `combo_descriptor_shuffled` AUC 0.570007 / logloss 0.736530
  - `no_gr_constant` AUC 0.500000 / logloss 0.877025
- `combo_descriptor_real` AUC gain:
  - vs shuffled: +0.089199
  - vs no-GR: +0.159206
- direct top1:
  - `combo_descriptor_real` RMSE 84.919128 / within10 0.560979
  - `no_gr_constant` RMSE 14.493051 / within10 0.691741

解釈:

- real GR shape descriptor は candidate-long likelihood feature として signal があり、negative controls を明確に上回る。
- ただし direct scorer / hard switch としては完全に不採用。`likpf_mean` 単体にも、no-GR constant の `pf_ancc` 固定にも大きく負ける。
- 後続に残す場合は exp092 系 ML confidence feature、または continuity-constrained verifier の feature 材料に限定する。

生成物 SHA:

- train feature raw SHA256: `e24a0803d2ade801e3bc655ea104df5e3042ef08b488e7489470ba379fed3e58`
- train feature decompressed SHA256: `e8d5fba94a6a9f0be401c023fc6b968c2e1dd5f3eeb40bacf98a2b262399cd4e`
- train feature schema SHA256: `ce378bd872ac2139dde2e1daa74e4122047a02caeed73048063d74bbeef46838`

記録更新:

- `config.yaml`: status を `completed_train_side_audit` に更新。
- `metrics.json`: logs 由来 summary で更新。
- `result.md`: 完了結果と解釈を記録。
- `KAGGLE_DIRECTION.md`: backlog から `gr_shape_descriptor_matching_ablation` を削除し、exp131 完了メモを追加する。
