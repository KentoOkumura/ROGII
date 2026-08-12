# 要件

## 依頼

exp378のformation-relative物理候補とtrain-only列・正解TVTから得た物理関係を、現在のPublic-LB-best参照exp335へadd-only特徴として利用する。fold mismatch leakageを防ぐstrict-nested設計を確定する。今回は実装・GPU学習を行わない。

## 制約

- Route: `ml_model`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp378のtechnical gateとcandidate novelty gateが合格するまで実装しない。
- 保存済みexp378 5-fold OOFをexp335へ直接流用しない。outer5×inner4でstrict-nested再生成する。
- exp335の370特徴、3 config、5 fold、学習設定を固定し、20特徴だけ追加する。
- 1 variant×3 config×5 fold=15 booster。親control再学習は0 booster。
- exp379/380/381のHMM/PF出力を本実験へ混ぜない。
- GPU push前に実行量を再提示し、明示承認を得る。

## 受け入れ基準

- nested 25 partitionすべてでrole auditが通り、outer-valid/testの生Formation列と正解TVT read countが0である。
- 特徴数が370+20=390で、追加列名・順序が設計と一致する。
- CV RMSEが8.096107755881022以下、5 fold中4 fold以上改善、3 LightGBM configすべて正である。
- H512/long401/clean273悪化が各0.02 ft以下、p95悪化0、worst悪化0.25 ft以下である。
- clean273悪化坑井数が+1/+3/+5 ftで150/53/21以下である。
- fold manifest、nested role表、20列schema/content SHA、model manifest/prediction SHA、kernel versionを記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
