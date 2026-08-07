# 要件

## 依頼

`KAGGLE_DIRECTION.md` の高優先度・基盤 backlog
`numba_allseed_pf_speed_reproduction` を `exp254` として実装する。

exp243 v3 の exact-parity likelihood-PF を reference とし、seed ごとの PF trajectory / log
likelihood を一度だけ生成する cold PF core と、保存済み seed bank を temperature、seed subset、
集約規則の多数 candidate に再合成する warm generation を分離して、速度・parity・決定性を監査する。

## 制約

- Route: `pf_beam`。
- PF dynamics、500 particles、seed 順、RNG、resampling、float64 dtype、入力 SHA は exp243 v3 を固定する。
- seed count は `1/4/16/32/64/128`、candidate spec count は `1/10/100/300` を事前固定する。
- temperature `3/5/8/12`、mean、prefix seed subset の再集約だけを warm candidate とする。
- single process / fixed Numba thread で legacy seed loop、Numba all-seed、cache 再集約を比較する。
- JIT compile、warm PF core、cache write/read、warm aggregation を別計時する。
- 代表 probe は exp243 対象 well の eval-length 分位から決定的に選ぶ。full workload は probe の全 guard 通過後だけ明示 mode で実行する。
- Python reference は wall-time budget で打ち切ってよいが、外挿値を実測値として記録しない。
- true TVT、error、oracle、Public/Private LB、score 改善、selector、exp218 feature、inference、submission を扱わない。
- 再現性: `docs/06_reproducibility.md` に従い、stable per-well seed、入力/出力 content SHA、反復 SHA、runtime metadata を記録する。

## 受け入れ基準

- exp243 reference、legacy seed loop、Numba all-seed の per-seed trajectory / log-likelihood / mean prediction が exact parity である。
- legacy / all-seed の repeated run と cache round-trip で SHA が一致する。
- seed count と candidate spec count の固定 grid を計時し、compile、PF core、cache I/O、warm aggregation を分離して CSV/JSON に保存する。
- peak RAM と測定 workload を記録し、773-well 外挿を実測値と明確に分ける。
- probe guard 不通過時は full workload が fail-closed する。
- LightGBM config 0、fold 0、booster 0、parent/control retraining なし、GPU なしである。
- notebook は Jupytext percent 形式から生成し、入力、設定、probe 選択、実行、guard、生成物をセル単位で追える。
- gzip 生成物を比較する場合は raw SHA ではなく decompressed content SHA を主証拠にする。本実験の seed bank cache は metadata 非依存の `.npz` payload content SHA を主証拠にする。
