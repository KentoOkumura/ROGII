# exp093_pf_candidate_coverage_then_ranker_audit セッションノート

## 目的

PF/Beam/likelihood-PF 候補集合に真値近傍候補が含まれる確率と oracle headroom を測り、supervised ranker / N-way classifier に進む前提があるかを確認する。coverage が低い bucket は ranker ではなく候補生成側の失敗として扱う。

## 現在の状態

- Route: pf_beam
- 状態: implemented_pending_kaggle_train
- CV: まだなし
- LB: なし
- 提出: なし

## コマンドログ

### 2026-06-20 JST 実装

```bash
make new-steering EXP=exp093_pf_candidate_coverage_then_ranker_audit
make new-exp EXP=exp093_pf_candidate_coverage_then_ranker_audit
```

実装内容:

- `.steering/20260620-exp093-pf-candidate-coverage-then-ranker-audit/` を作成し、requirements / design / tasklist を記入した。
- `config.yaml` を train-side candidate coverage audit 用に更新した。
- `pf_candidate_coverage_then_ranker_audit.py` を追加し、exp091 の self-GR 候補生成を再利用しつつ、candidate set 別 oracle coverage、bucket coverage、ranker readiness 判定を実装した。
- train notebook を、設定確認、入力前提、監査実行、出力 preview、metrics 保存のセル構成に更新した。
- inference notebook は診断専用 no-op として明記した。

### 予定

```bash
make validate-exp EXP=exp093_pf_candidate_coverage_then_ranker_audit
make prepare-kaggle-notebooks EXP=exp093_pf_candidate_coverage_then_ranker_audit EXTRA_ARGS="--notebook train --run-on-push --strict"
make push-kaggle-train EXP=exp093_pf_candidate_coverage_then_ranker_audit
```

### 2026-06-20 JST Kaggle train v1 push

最初の push は default title が kernel id に slug 解決されず、Kaggle API 400 で失敗した。

```bash
kaggle kernels push -p experiments/exp093_pf_candidate_coverage_then_ranker_audit/kaggle/train
```

失敗理由:

```text
Your kernel title does not resolve to the specified id.
```

短い canonical id / title で train package を再生成した。

```bash
make prepare-kaggle-notebooks EXP=exp093_pf_candidate_coverage_then_ranker_audit EXTRA_ARGS="--notebook train --kernel-id kentookumura/exp093-pf-candidate-coverage-ranker-train --title 'exp093 pf candidate coverage ranker train' --run-on-push --strict"
kaggle kernels push -p experiments/exp093_pf_candidate_coverage_then_ranker_audit/kaggle/train
```

結果:

- Kernel version 1 push 成功。
- URL: https://www.kaggle.com/code/kentookumura/exp093-pf-candidate-coverage-ranker-train
- `kaggle kernels pull kentookumura/exp093-pf-candidate-coverage-ranker-train -p /tmp/kaggle-pull/exp093-pf-candidate-coverage-ranker-train -m` で存在確認済み。
- 通常 `logs` は初回取得時点では空。
- `logs -f --interval 20` はユーザー指示によりこちら側の監視だけ停止。Kaggle kernel 実行自体は停止していない。

### 2026-06-20 JST Kaggle train v1 output 取得

ユーザー完了連絡後に logs / output を取得した。

```bash
kaggle kernels logs kentookumura/exp093-pf-candidate-coverage-ranker-train
kaggle kernels output kentookumura/exp093-pf-candidate-coverage-ranker-train -p experiments/exp093_pf_candidate_coverage_then_ranker_audit/kaggle/output/train_v1
```

結果:

- status: `completed_train_side_audit`
- rows: 3,783,989
- wells: 773
- runtime: 3,851.92 sec
- best single candidate: `likpf_mean` RMSE 11.594897 / within10 0.772807
- baseline oracle: RMSE 7.434030 / within10 0.906525
- baseline+self-GR oracle: RMSE 6.958935 / within10 0.922492
- target-free rank score top1 は baseline+self-GR で RMSE 29.985529 と弱い
- recommendation: `ranking_or_likelihood_scorer_audit_before_ranker`

追加で candidate long から候補ごとの oracle best / rank score top1 選択回数を集計した。

```bash
.venv/bin/python - <<'PY'
# candidate_long.csv.gz を chunksize で読み、候補ごとの oracle best と rank score top1 を集計
PY
```

保存先:

- `kaggle/output/train_v1/artifacts/exp093_pf_candidate_coverage_then_ranker_audit_selection_counts.csv`

要点:

- 完全なノイズ候補、つまり rank score top1 も oracle best も 0 の候補はなかった。
- `pf_ancc` は oracle best 1,092,069 rows だが rank score top1 は 0 rows。
- `self_gr_sc25`、`self_gr_best`、`self_gr_sc15` も oracle best になる場面はあるが、rank score top1 では選ばれていない。
- 現行 rank score は `likpf_mean` と `beam_mean` に偏っており、PF ANCC と self-GR scale candidates の headroom を活かせていない。

## 生成予定ファイル

- `exp093_pf_candidate_coverage_then_ranker_audit_candidate_metrics.csv`
- `exp093_pf_candidate_coverage_then_ranker_audit_rank_metrics.csv`
- `exp093_pf_candidate_coverage_then_ranker_audit_bucket_metrics.csv`
- `exp093_pf_candidate_coverage_then_ranker_audit_candidate_set_bucket_metrics.csv`
- `exp093_pf_candidate_coverage_then_ranker_audit_by_well.csv`
- `exp093_pf_candidate_coverage_then_ranker_audit_self_gr_well_summary.csv`
- `exp093_pf_candidate_coverage_then_ranker_audit_row_context.csv.gz`
- `exp093_pf_candidate_coverage_then_ranker_audit_candidate_long.csv.gz`
- `exp093_pf_candidate_coverage_then_ranker_audit_feature_schema.csv`
- `exp093_pf_candidate_coverage_then_ranker_audit_summary.json`
- `exp093_pf_candidate_coverage_then_ranker_audit_selection_counts.csv`

## 再現性メモ

- seed policy: exp093 内では新規乱数なし。
- stochastic components: upstream exp072 PF/Beam cache のみ。exp093 では再生成しない。
- CPU/GPU runtime: CPU train-side audit。GPU 不要。
- Kaggle kernel id / version: `kentookumura/exp093-pf-candidate-coverage-ranker-train` v1 pushed。
- input / feature schema SHA: Kaggle train 完了後に summary JSON へ記録予定。
- feature content SHA: source cache の gzip decompressed content SHA を主証拠にする。
- model manifest / model SHA: model なし。
- prediction SHA: prediction なし。
- submission SHA: submission なし。
- rerun check: deterministic submission anchor ではないため対象外。

## 次のアクション

1. ユーザーから Kaggle 実行完了連絡を受けたら、logs / output を取得する。
2. summary JSON の recommendation を `result.md`、`metrics.json`、`experiment_summary.md` に記録する。
