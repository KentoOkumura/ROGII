# exp423_same_typewell_gr_dtw_truth_warp_transfer_readout セッションノート

## 目的

same-typewell 内の GR 類似 donor から正解 TVT warp を query well へ転写する仮説を、
leak-free な 0-model Stage 0 として設計・実装する。

## 現在の状態

- Route: `pf_beam`
- 状態: Kaggle CPU audit 完了・gate FAIL・branch 閉鎖
- CV: `14.103812714`
- LB: 対象外
- 正規 train notebook: compact self-contained 版を採用
- 正規 inference notebook: scaffold placeholder（対象外）
- compact self-contained train notebook: 実装済み
- package / Kaggle run: 完了
- inference / submission: 対象外・未実施

## 2026-07-28 設計セッション

### 実行済みコマンド

```bash
task new-steering EXP=exp423_same_typewell_gr_dtw_truth_warp_transfer_readout
```

`task` executable が環境に無かったため失敗した。Makefile fallback を使用した。

```bash
make new-steering EXP=exp423_same_typewell_gr_dtw_truth_warp_transfer_readout
make new-exp EXP=exp423_same_typewell_gr_dtw_truth_warp_transfer_readout
```

### 確定した設計

- parent: `exp109_typewell_neighbor_prior_features`
- route: `pf_beam`
- 5-fold outer-valid pseudo-tail readout
- same-typewell / outer-train donor pool
- 256-point robust-normalized GR constrained-DTW
- top-K=5、primary=`analog_top5_median`
- donor truth delta の query anchor への転写
- query truth late join、donor/query intersection zero
- stable random control、top-5 per-well oracle headroom
- fixed AND gates、rescue grid なし

### 実装量

- variant 数: 0
- LightGBM config 数: 0
- fold 学習数: 0
- booster 数: 0
- PF/HMM/Beam 実行数: 0
- baseline/control 再学習: なし

## 再現性メモ

- seed policy: matching は乱数なし。negative control のみ stable SHA256。
- stochastic components: なし。
- CPU/GPU runtime: CPU-only、single process、GPU/AMP なし。
- query truth: artifact freeze 前 read count 0 を必須 assert。
- donor truth: outer-train donor path の materialization に限って使用。
- SHA: input、row inventory、config、schema、decompressed feature/prediction content を記録予定。
- model SHA / submission SHA: model と submission を作らないため対象外。
- deterministic anchor: 未認定。実装後の独立再実行一致が必要。

## 2026-07-28 実装セッション

ユーザーの `exp423を実装してください` を実装承認として扱い、正規 notebook を
上書きせずに次を追加した。

- `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_compact_selfcontained_train.py`
- `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_compact_selfcontained_train.ipynb`
- `experiments/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout/tests/test_exp423_same_typewell_gr_dtw_truth_warp_transfer_readout.py`

実装した処理:

1. exp099 の 3,783,989-row inventory、exp065 `native_overlap=1` group、
   exp109 固定 best を SHA 検証付きで target-free に読む。
2. exp109 と同じ sorted-well + seed 42 の 5-fold を再現し、query/outer-valid
   horizontal は `MD/GR/TVT_input` だけを読む。
3. suffix ごとの centered rolling median、256-point progress 補間、robust
   z-normalization、support guard を実装する。元 GR 欠損位置は smoothing 後も
   unsupported とし、端点外挿しない。
4. Sakoe-Chiba band 32、axis-run 上限 4 の Numba constrained-DTWを実装し、
   `(cost, donor well_id)` で固定順位付けする。
5. fold 内 outer-train donor の `TVT` だけを materialize し、query anchor へ
   donor delta を再アンカーする。top-1、top-5 median、stable random を freeze する。
6. candidate、全 top-5 donor path、DTW path、fold separation、input/config/schema/
   logical/decompressed SHAを query truth 前に保存する。
7. freeze 後だけ query truth を読み、whole-well oracle、overall/fold/1000+/
   hidden-like/by-well、DTW cost-error Spearman、固定 AND gate を計算する。
8. 初回 run は独立 rerun reference SHA がないため determinism gate を
   `pending_independent_rerun_reference` とし、結果を勝手に deterministic anchor
   へ昇格しない。

親 exp109 には compact self-contained source がないため、親 compact との章比較は
対象外。exp423 は 11 章 / 2,000 行超で、同一 exp helper import だけの薄い notebook
ではない。compact source に `__file__` は残していない。

実行量:

- audit variant: 1
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- PF / HMM / Beam well-run: `0 / 0 / 0`
- GPU run: 0
- baseline/control 再学習: なし
- reporting fold: 5

検証コマンド:

```bash
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb experiments/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_compact_selfcontained_train.py
JUPYTER_DATA_DIR=/tmp/jupyter-data .venv/bin/jupytext --to ipynb --test experiments/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_compact_selfcontained_train.py
.venv/bin/python -m py_compile experiments/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_compact_selfcontained_train.py
.venv/bin/ruff check experiments/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout_compact_selfcontained_train.py experiments/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout/tests/test_exp423_same_typewell_gr_dtw_truth_warp_transfer_readout.py --select F821
.venv/bin/pytest -q experiments/exp423_same_typewell_gr_dtw_truth_warp_transfer_readout/tests/test_exp423_same_typewell_gr_dtw_truth_warp_transfer_readout.py
make validate-exp EXP=exp423_same_typewell_gr_dtw_truth_warp_transfer_readout
```

この時点では Kaggle package、正規 notebook 採用、CPU audit run、inference、
submission は行っていない。

## 2026-07-28 実行承認

ユーザーの `実行してください` を、正規 train notebook 採用、Kaggle package、
Kaggle CPU audit の push / 実行承認として記録した。inference と submission は
承認範囲外のままとする。

push 前の固定実行量:

- audit variant: 1
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- PF / HMM / Beam well-run: `0 / 0 / 0`
- GPU run: 0
- baseline/control 再学習: なし
- reporting fold: 5

初回 package は長すぎる kernel slug/title により Kaggle API 400 となり、run は
作成されなかった。短い正規 slug
`kentookumura/exp423-gr-dtw-truth-warp-readout-train` に変更して version 1 を
push したところ、旧表記の exp099 kernel source
`exp099-pf-multi-observation-likelihood-probe-train` が無効として除外された。
Kaggle 検索で実在 slug が `exp099-pf-multiobs-likelihood-train` であることを確認し、
config の kernel source と Kaggle input candidate を修正した。科学設定、variant 数、
fold 数、booster 数は変更していない。

### Kaggle version 2: 初回有効 run

- kernel: `kentookumura/exp423-gr-dtw-truth-warp-readout-train`
- status: `COMPLETE`
- runtime: 772.01 秒
- rows / wells: `3,783,989 / 773`
- supported rows / wells: `1,394,464 / 286`
- supported score-row / well fraction: `0.368517 / 0.369987`
- target-free logical content SHA:
  `6b5b54521ba6612665436f95d4ab3d42c711e8eb18a29bb2ad1916862849d3b3`
- query truth rows before / after freeze: `0 / 3,783,989`
- donor/query intersection: 全 5 fold で 0
- primary RMSE: `14.103813`
- exp109 fixed RMSE: `11.143367`
- primary gain vs exp109: `-2.960446 ft`
- top-5 per-well oracle RMSE / gain: `12.285086 / -1.141720 ft`
- top-1 gain vs stable random: `+1.233004 ft`
- pooled DTW cost-error Spearman: `0.102226`
- scientific gate: FAIL
- technical gate: support coverage 2 checksがFAIL、determinismは独立rerun待ち

oracle headroom 自体が負で、primary は全 5 fold、1000+、hidden-like で exp109 より
悪化した。固定設計に従って parameter rescue は行わない。再現性契約を完了するため、
上記 logical content SHA を `expected_target_free_content_sha256` に固定し、
同一科学契約の独立 rerun を 1 回だけ行う。

### Kaggle version 3: 独立 rerun

- status: `COMPLETE`
- runtime: 786.57 秒
- scientific contract SHA:
  `0429bae5f1cb16cc209a4a9e50fdccef1a62ce505591a265115e65bd33c148ca`
- observed / expected logical content SHA:
  `6b5b54521ba6612665436f95d4ab3d42c711e8eb18a29bb2ad1916862849d3b3`
- determinism: `matched_independent_reference`
- version 2 / 3 で scope、fold、by-well、donor-rank、Spearman、
  fold-separation、scientific-gate artifact の SHA が一致
- technical gate: deterministic check は PASS、coverage 2 checks は同じく FAIL
- scientific gate: version 2 と同値で FAIL

`kaggle/train/` は実行した version 3 payload の証跡として保持するため、埋め込み
config の status は `independent_rerun_ready`、metadata は `run_on_push=true` のまま
である。実行完了後の正規 `config.yaml` / `metrics.json` は
`completed_closed` に更新しており、この package を再 push しない。

### 最終判断

fixed branch rule の oracle 不合格に該当する。same-typewell の個別 donor
truth-warp transfer は group 平均 prior を上回る headroom がなく、primary は
全 5 fold、1000+、hidden-like で悪化したため、PF/Beam candidate、inference、
submission へ進めず閉じる。top-1対randomの正方向だけで top-K、band、support、
group、selector を same-OOF 救済しない。

## 次のアクション

exp423 の追加実行は行わない。GR similarity を補助品質特徴として使う既存 backlog
候補は、truth-warp correction と分離し、今回の negative transferability と低 coverage
を前提に再評価する。inference / submission への昇格はしない。
