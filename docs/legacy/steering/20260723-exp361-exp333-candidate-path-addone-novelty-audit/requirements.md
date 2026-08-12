# 要件

## 依頼

exp333 を exp228 / exp263 の単体置換候補としてだけ扱わず、exp293 で固定した
12候補バンクへ追加する候補パスとして再評価する。exp302 と同じ add-one
novelty 契約を使い、exp333 の current-test 推論を実装する価値があるかを判定する。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp333 Stage 1 の保存済み OOF だけを追加候補として使い、再学習・補正・blend・候補重み探索をしない。
- 固定12候補バンク、候補順、float32 化、H128/H256/H512/whole-well block を exp293/exp302 から変更しない。
- exp226 は直接性能の一次 control、exp228 は row-wise residual の参考 ablation、exp263 は固定 blend の参考値に限定し、novelty の hard gate にしない。
- 評価 truth を読む前に、固定バンク、block assignment、exp333 target-free OOF、入力契約を SHA 付きで freeze する。
- この実験では raw-test inference と submission を実施しない。PASS 後も exp333 内での inference 実装は別承認とする。
- Kaggle CPU notebook 1回、候補1本、評価5 fold、LightGBM config 0、trained fold 0、booster 0、GPU 0 とする。

## 受け入れ基準

- technical guard がすべて PASS する。
- H512 add-one oracle RMSE 改善が `0.03 ft` 以上。
- whole-well add-one oracle RMSE 改善が `0.02 ft` 以上。
- H512 strict unique-best block fraction が `2%` 以上。
- H512 oracle RMSE が 5 fold 中 4 fold 以上で改善する。
- PASS の場合は `exp333_candidate_path_novelty_supported`、FAIL の場合は `close_exp333_candidate_novelty_branch` と判定する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
