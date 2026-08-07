# 要件

## 依頼

`KAGGLE_DIRECTION.md` の backlog `multi_scale_initial_rate_candidates` を
`exp268_multi_scale_initial_rate_candidates` として実装する。exact HMM が prefix 末尾30行から
推定する単一初期rateへの依存を切り分けるため、known prefixだけから固定window
`32/64/128/256` のrobust slopeを作り、それぞれを独立したHMM candidateとして保持する。

## 制約

- Route: `pf_beam`。
- 親は `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation` とし、保存済み
  `tail_n=30` HMMをcontrolとして再利用する。control HMMは再生成しない。
- HMMのgrid、rate grammar、transition、GR emission、calibration、missing-GR処理はexp209から固定し、
  変更するのは初期rate推定windowだけにする。
- window rateは `median((delta TVT_input + delta Z) / delta MD)` とし、evaluation tailのtrue TVT、
  error、oracle、Public LBを計算やwindow選択に使わない。
- base `tail_n=30` candidateを必ずbankに残す。`32/64/128/256` candidateの平均、blend、
  target-aware選択、tail中のprior pull、追加rate推定器、weight学習は実装しない。
- row / 128・256・512-row block / whole-well oracleは、candidateを固定した後の診断に限定する。
- 4候補の全well実行はKaggle timeout回避のためtarget-free hashで2 well shardに分割し、
  aggregate notebookでcoverageと重複をfail-closedに検証する。
- 学習0、LightGBM config 0、fold 0、booster 0、GPU 0、raw-test inference 0、submission 0。
- 再現性: `docs/06_reproducibility.md` に従い、Kaggle bootstrap、入力SHA、shard出力のraw/decompressed
  SHA、aggregate prediction content SHA、kernel versionを記録する。

## 受け入れ基準

- `config.yaml` に `experiment.route=pf_beam`、親、4 window、2 shard、HMM固定値、実行コストがある。
- compact self-contained Jupytext notebookとして、2 shard generator、aggregate train、disabled inferenceを
  実装し、同一実験helper importと`__file__`を使わない。
- shardごとに4候補、initial rate、effective prefix rows、HMM std/loglikを保存し、well shardは
  stable SHA256のみで決める。
- aggregateはexp209 `tail_n=30` control、exp072 `likpf_mean` reference、4新規候補をidでstrict alignする。
- overall、distance bucket、prefix-length、hidden-like、by-well/worst-well、rate spread、rate/path重複率、
  unique-best、row/block/whole-well oracleを出力する。
- true TVT/error/oracleがcandidate生成・候補選択へ流れないcontract testがある。
- Jupytext round-trip、py_compile、Ruff F821、固有test、`make validate-exp`、`make validate-template`が通る。
- deterministic anchorとは扱わず、gzipはdecompressed content SHAを主証拠として記録する。
