# 要件

## 依頼

`KAGGLE_DIRECTION.md` の `pf_ancc_small_seed_mean_candidate_audit` を
`exp271_pf_ancc_small_seed_mean_candidate_audit` として実装する。

exp266 で固定済みの PF ANCC seed 順の先頭 4 / 8 seed を同じ kernel・600 particles で
全 train pseudo-tail へ再生成し、mean path を exp263 core 12 candidate bank に追加する価値を
0-booster の oracle / stability readout で監査する。

## 仮説

exp266固定seed順の先頭4 seed meanが8 seed meanと同程度のexp263 core-bank追加headroomを持てば、
64 seed再生成より低コストなPF ANCC candidateとして後続監査に残せる。

## 親・変更点

- 親: `exp266_pf_ancc_pf_z_multiseed_stability_audit`
- 比較bank: `exp263_last_anchor_better_candidate_confidence_pair_cache` core 12
- 変更: PF ANCCの固定先頭4/8 seed meanを保存し、candidate追加headroomだけを測る。
- 固定: PF kernel、600 particles、seed順、mean集約、candidate bank、audit scope。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- seed 0 は exp072/exp266 の `stable_seed("pf_ancc", well)`、seed 1〜7 は
  `stable_seed("exp266_pf_ancc_pf_z_multiseed_stability_audit", "train", "pf_ancc", well, seed_index)`
  をそのまま使う。
- exp266 の true TVT 結果、strong well ID、original-seed percentile、error、oracle を
  PF generation、seed 選択、candidate gate に使わない。
- PF ANCC dynamics、600 particles、resampling、先頭 seed 順、mean 集約を固定し、grid 化しない。
- PF-Z、selector 学習、hard routing、direct inference、submission は対象外とする。
- exp263 core 12 candidate bank は保存済み Stage 0 cache を読み、再生成・再学習しない。
- LightGBM config / fold / booster は `0 / 0 / 0`、parent/control retraining は行わない。
- Kaggle Notebook の初回フル実行を正とし、ローカル notebook 実行は行わない。

## 受け入れ基準

- 3,783,989 rows / 773 wells を対象に、4-seed / 8-seed PF ANCC mean path と
  row-wise seed disagreement を保存できる。
- seed 0 path が exp072 `pf_ancc` と全行 exact parity、4/8-seed per-well RMSE が
  exp266 保存済み集約と許容差内で一致する fail-closed guard を持つ。
- exp263 core 12 に対し、bank only / +mean4 / +mean8 / +both の row、block 128/256/512、
  whole-well oracle RMSE、unique-best、distance bucket、hidden-like、worst-well を比較する。
- 4/8 単体 RMSE、4-vs-8 差、seed disagreement と誤差/unique-headroom の readout、実測 runtime を保存する。
- candidate path gzip は raw gzip SHA と decompressed content SHA を分け、後者を主証拠にする。
- deterministic submission anchor とは扱わず、model SHA / submission SHA は非該当として明記する。
- `config.yaml`、Jupytext train/inference、notebook、`SESSION_NOTES.md`、`README.md`、
  `result.md`、`metrics.json` を実験ディレクトリに揃える。
- py_compile、Ruff F821、Jupytext round-trip、実験 validation、対象 unit test が通る。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 次のアクション

実装完了後はKaggle CPU実行を別途明示依頼された場合だけ行い、4/8差とguardから縮約・閉鎖を判断する。
