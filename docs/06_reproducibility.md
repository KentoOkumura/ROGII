# 再現性ガード

この repo では、Kaggle Notebook 上で同じ実験を再実行したときに、少なくとも採用候補の `submission.csv` が説明可能な範囲で固定されることを再現性の基準にします。CV、OOF、生成特徴、submission のどこが固定され、どこが環境差で揺れるかを分けて記録します。

## 基本方針

- seed、fold、feature schema、入力生成物、model config、runtime CPU/GPU、kernel source version を `config.yaml` と `SESSION_NOTES.md` に残す。
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

`prepare-kaggle-notebooks` が作る Kaggle notebook には、`config.yaml`、補助 `.py`、`project.yml`、`src/` を復元する bootstrap ZIP が埋め込まれる。生成後に `kaggle/<kind>/config.yaml` や補助 `.py` を手で直しただけでは、Kaggle 実行時の notebook 先頭セルが古い内容を展開することがある。

運用ルール:

- 原則として正の編集対象は `experiments/<exp>/config.yaml`、`<exp>_train.ipynb`、`<exp>_inference.ipynb`、補助 `.py` とし、編集後は `prepare-kaggle-notebooks` を再実行する。
- CPU/GPU など派生 package を手で作る場合は、`kernel-metadata.json` だけでなく notebook bootstrap ZIP 内の support files も同じ設定になっていることを確認する。
- push 前に、生成 notebook の bootstrap ZIP から `config.yaml` を取り出して `selected_mode`、`kernel_sources`、`enable_gpu`、seed 設定が期待通りか確認する。
- v1 が設定不整合で失敗した場合は、同じ canonical kernel id に v2 として再 push し、原因、修正、失敗した version を `SESSION_NOTES.md` と `metrics.json` に残す。

## 記録する証拠

再現性を主張する実験では、少なくとも次を `metrics.json` または `SESSION_NOTES.md` に残す。

- Kaggle kernel id、version、URL、kernel source id、runtime CPU/GPU、internet disabled。
- 入力 cache / artifact の file SHA、schema SHA、row count、well count、feature count。
- gzip 出力を比較する場合は raw gzip SHA と decompressed content SHA を分ける。
- model manifest の model count、各 model SHA、selected mode、selected model。
- OOF prediction SHA、test prediction content SHA、submission SHA。
- `submission.csv` の submit-check 結果、fallback rows、prediction min/max/mean/std。
- GPU と CPU を比較した場合は、CV 差分と submission 差分の abs mean / abs max / mean。
- rerun した場合は、version ごとの feature content SHA、prediction SHA、submission SHA、byte-identical 判定。

## 採用判断

- `submission.csv` が複数 rerun で byte-identical、または差分が説明済みで submission SHA が固定されている場合だけ deterministic submission anchor とする。
- CV だけが再現していても、hidden test feature regeneration が stochastic なら deterministic anchor ではない。
- Public LB が良くても、feature SHA / submission SHA が固定されていない候補は stochastic candidate として扱う。
- ML route の CV anchor と PF/Beam route の deterministic replay candidate は、根拠が違うため同じ強さの anchor として混ぜない。

## 提出前チェック

- `task validate-exp EXP=<exp>` が通る。
- Kaggle package の metadata と bootstrap 内 config が一致している。
- `kaggle kernels pull <kernel> -p /tmp/kaggle-pull/<slug> -m` で同じ kernel id の存在を確認している。
- output を取得し、`task submit-check` または `scripts/validate_submission.py` が通っている。
- `SESSION_NOTES.md`、`result.md`、`metrics.json`、`experiment_summary.md` に、command、version、SHA、解釈、次アクションが揃っている。
