# 再現性ガード

この repo では、Kaggle Notebook 上で同じ実験を再実行したときに、少なくとも採用候補の `submission.csv` が説明可能な範囲で固定されることを再現性の基準にします。CV、OOF、生成特徴、submission のどこが固定され、どこが環境差で揺れるかを分けて記録します。

## 基本方針

- 実験記録全体の役割分担は`AGENTS.md`を正とする。再現性については、実行前に決めるseed、fold、feature schema、入力生成物、model configを設定として残し、実行後に確定するruntime情報とSHAを構造化された証拠として、取得過程を時系列ログとして記録する。
- stochastic な処理は global RNG に依存させず、`np.random.default_rng(seed)` のような局所 RNG を渡す。
- `joblib.Parallel(... prefer="threads")` や thread pool 内で global RNG を使わない。並列順序で乱数消費が変わるため、well id、fold id、variant 名などの immutable key から stable seed を作る。
- PF/Beam、likelihood-PF、DTW sampling、seed bagging など候補生成が stochastic な実験では、per-well stable seed を必須にする。難しい場合は deterministic mode として `n_jobs=1` を用意し、その制約を記録する。
- train と inference の feature generation は別物として監査する。train cache が deterministic でも、code competition の hidden test inference が raw test から stochastic に再生成されるなら submission は固定されない。
- GPU 学習は bitwise reproducible と決めつけない。LightGBM GPU を使う場合は `gpu_use_dp=true`、`deterministic=true`、`force_col_wise=true`、固定 `num_threads` / `n_jobs` を検討し、必要なら CPU deterministic control も 1 回作る。
- inference は保存済み booster / model artifact を読む。毎回再学習して submission を作る flow は、学習と推論を分けて SHA を追える形にする。

## PF/Beam と raw-test regeneration

PF/Beam 系の実験では、次を満たさない限り deterministic anchor と呼ばない。

- PF/Beam/likelihood-PF の乱数 seed が、`well id`、`split`、`feature family`、`seed index` などから SHA256 等で安定生成される。
- global `np.random.randn`、`np.random.uniform`、`random.random` を並列処理内で直接使わない。
- thread scheduling に依存して結果が変わらない。並列化しても各 well の乱数系列が独立している。
- raw train 由来 cache と raw test 再生成の両方について、feature row count、well count、feature count、schema、content SHA を記録する。
- raw `.csv.gz` の SHA は gzip metadata や書き出し条件で変わることがあるため、feature determinism の主証拠にはしない。gzip は decompressed CSV content SHA を主証拠にする。

## Kaggle package bootstrap

`prepare-kaggle-notebooks` が作る Kaggle notebook には、`config.yaml`、`metrics.json`、補助 `.py`、`project.yml`、`src/` を復元する bootstrap ZIP が埋め込まれる。`metrics.json`はNotebook側の部分更新で既存のstatus、CV/LB、実行証拠を保持するために含める。生成後に `kaggle/<kind>/config.yaml`、`metrics.json`、補助 `.py` を手で直しただけでは、Kaggle 実行時の notebook 先頭セルが古い内容を展開することがある。

運用ルール:

- 原則として正の編集対象は `experiments/<exp>/config.yaml`、`<exp>_train.ipynb`、`<exp>_inference.ipynb`、補助 `.py` とし、編集後は `prepare-kaggle-notebooks` を再実行する。
- CPU/GPU など派生 package を手で作る場合は、`kernel-metadata.json` だけでなく notebook bootstrap ZIP 内の support files も同じ設定になっていることを確認する。
- push 前に、生成 notebook の bootstrap ZIP から `config.yaml` と `metrics.json` を取り出し、設定、既存status、CV/LB、実行証拠が期待どおりか確認する。runtime resourceは`kernel-metadata.json`の`enable_gpu`と`enable_tpu: false`で確認する。
- v1 が設定不整合で失敗した場合は、同じ canonical kernel id に v2 として再 pushする。原因、修正、再実行コマンドは`SESSION_NOTES.md`、失敗・成功したkernel versionと生成物SHAは`metrics.json`へ分担して残す。

## 記録する証拠

再現性を主張する実験では、少なくとも次を`metrics.json`の`evidence`へ機械可読に残す。値を得たコマンド、時刻、途中経過だけを`SESSION_NOTES.md`へ記録する。

- Kaggle kernel id、version、URL、kernel source id、CPU/GPUなどのresource、Notebook実行時間、internet enabled/disabled。
- 入力 cache / artifact の file SHA、schema SHA、row count、well count、feature count。
- gzip 出力を比較する場合は raw gzip SHA と decompressed content SHA を分ける。
- model manifest の model count、各 model SHA、selected mode、selected model。
- OOF prediction SHAは`oof_prediction_sha`、test prediction content SHAは`test_prediction_content_sha`、submission SHAは`submission_sha`へ分けて記録する。意味が曖昧な汎用`prediction_sha`は使わない。
- `submission.csv` の submit-check 結果、fallback rows、prediction min/max/mean/std。
- GPU と CPU を比較した場合は、CV 差分と submission 差分の abs mean / abs max / mean。
- rerun した場合は、`evidence.reruns`の各要素を次の形で記録する。`reference_submission_sha`は比較元、`submission_sha`はそのrerunの値とする。差分がある場合は`difference_notes`を必須にする。

```json
{
  "kernel_version": 2,
  "feature_content_sha": "...",
  "test_prediction_content_sha": "...",
  "submission_sha": "...",
  "reference_submission_sha": "...",
  "byte_identical_to_reference": true,
  "difference_notes": null
}
```

新規実験は`templates/experiment/metrics.json`の`evidence` schemaを使う。template notebookは`settings.py`の`update_metrics`で実行が所有する値だけを部分更新し、既存のLBやsubmission証拠を消さない。既存フィールドを保ったまま個別の証拠を記録する場合は、`record-exp`の`--evidence`を必要な回数だけ指定する。keyは`evidence`直下からの相対pathとし、数値、boolean、list、objectはJSONとして解釈される。

```bash
task record-exp EXP=expXXX_title EXTRA_ARGS="--evidence kaggle.kernel_id=owner/slug --evidence kaggle.kernel_version=1 --evidence kaggle.notebook_runtime_seconds=3600"
```

## 再現性上の提出候補条件

ここでは再現性の観点から提出候補として扱える条件だけを定めます。実験の完了、採用、不採用はこれらの条件だけで確定せず、比較結果と未解決事項を整理したうえでユーザーが判断します。

- `submission.csv` が複数 rerun で byte-identical、または差分が説明済みで submission SHA が固定されている場合だけ deterministic submission anchor とする。
- CV だけが再現していても、hidden test feature regeneration が stochastic なら deterministic anchor ではない。
- Public LB が良くても、feature SHA / submission SHA が固定されていない候補は stochastic candidate として扱う。
- ML route の CV anchor と PF/Beam route の deterministic replay candidate は、根拠が違うため同じ強さの anchor として混ぜない。

## 提出前チェック

- `task validate-exp EXP=<exp>` が通る。
- Kaggle package の metadata と bootstrap 内 config が一致している。
- `uv run kaggle kernels pull <kernel> -p /tmp/kaggle-pull/<slug> -m` で同じ kernel id の存在を確認している。
- 実ファイルの検証が必要な場合はoutputを取得し、`task submit-check EXP=<exp> SUBMISSION=<path>`が通り、検証結果とsubmission SHAが対象実験の`metrics.json`へ保存されている。`task`が利用できない環境では同じ変数で`make submit-check`を使う。ローカルに取得しないcode submissionは、Kaggle上の`submission.csv`の存在とnotebook内の形式検証結果を記録する。
- command、version、SHA、解釈、次アクションが`AGENTS.md`で定めた正本へ分担され、同じ情報を複数ファイルへ手作業で転記していない。
