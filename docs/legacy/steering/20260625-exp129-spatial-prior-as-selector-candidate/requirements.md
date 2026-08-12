# 要件

## 依頼

`spatial_prior_as_selector_candidate` バックログを実装する。exp114 の fold-safe spatial prior TVT を exp099/101 系の PF/Beam candidate selector surface に候補 path として追加し、raw replacement ではなく「選ぶ価値がある候補か」を train-side OOF で診断する。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- validation/test true TVT を candidate 生成や selector feature source に入れない。
- exp099 v2 candidate cache と exp114 v1 OOF spatial prior artifact は固定入力として扱う。
- spatial prior は soft average で混ぜず、候補 path として選ぶか選ばないかを評価する。
- Kaggle CPU Notebook を正とし、GPU は使わない。
- positive result でも direct submit には進めず、raw-test/full-train parity と hidden-like stress を次段で確認する。

## 受け入れ基準

- `.steering` に仮説、固定入力、leakage 制約、再現性設計が記録されている。
- `experiments/exp129_spatial_prior_as_selector_candidate/` に config、補助コード、train/inference notebook、記録ファイルがある。
- train notebook で exp099 cache と exp114 OOF artifact を結合し、base 5候補 + spatial 2候補の oracle、topK、candidate metrics、selection distribution、by-well continuity、bucket metrics を保存できる。
- predicted-error ranker と Viterbi smoothing grid を OOF で比較できる。
- deterministic anchor ではない train-side audit として、入力 SHA、model SHA、prediction SHA、Kaggle kernel version が記録される。submission SHA は生成しない。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
