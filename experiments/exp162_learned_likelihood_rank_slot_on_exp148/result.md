# exp162 結果

## 結論

CPU full 15 boosters の単一 notebook は timeout したため、`lgb0` / `lgb1` / `lgb2` を別 Kaggle train notebook に分割して再実行した。3本とも `KernelWorkerStatus.COMPLETE` まで完了し、各 split の model manifest / OOF predictions / feature importance / by-well / bucket metrics を生成した。

split train の pooled RMSE は `lgb0` 8.488049241、`lgb1` 8.456600574、`lgb2` 8.443346041。`lgb1` / `lgb2` は exp148 `lgb_mean` CV 8.501281182 より良い。cross-manifest の OOF `lgb_mean` はまだローカル集計していないため、正式な採用判断では split OOF predictions を使った追加確認が残る。

inference v4 は hidden-safe 修正版として完了した。`exp145-inference` の public raw-test features 依存を外し、public / hidden とも current-test learned-likelihood features を生成する経路で実行した。3つの split train manifest から合計 15 booster を読み込み、`submission.csv` を生成した。submit-check は PASS。code submit ref `54247043` は Public LB `8.100` で、exp148 `7.960` より +0.140 悪化したため採用しない。

## 比較基準

- exp148 `lgb_mean` CV: 8.50128118189582
- exp148 Public LB: 7.960
- exp092 `lgb1` CV: 9.322479895503927
- exp098 `lgb1` CV: 9.35815105231876
- exp153 `lgb_mean` CV: 9.423385453890534

## Kaggle split train metrics

- `lgb0`: pooled RMSE 8.48804924068525
- `lgb1`: pooled RMSE 8.456600573816607
- `lgb2`: pooled RMSE 8.443346041268544
- split manifest ensemble `lgb_mean`: inference 側では3 manifest の全 booster を平均する。OOF ensemble は未集計。

## 次アクション

exp162 は Public LB で exp148 を更新できなかったため、ML route anchor は exp148 のまま維持する。必要なら split OOF predictions を取得して、CV 改善が LB に転移しなかった原因を by-well / bucket / cross-manifest `lgb_mean` で確認する。

## Inference v3

- kernel: `kentookumura/exp162-ll-rank-slot-exp148-infer` v3
- status: `KernelWorkerStatus.COMPLETE`
- selected model: `lgb_mean`
- loaded boosters: 15
- test rows / submission rows: 14151 / 14151
- fallback rows: 0
- prediction range: 11590.8623046875 - 12240.15234375
- prediction mean / std: 11905.147258047686 / 278.86038280580544
- prediction SHA256: `2c93fe0030206d0d9824edb368c72002730868a3ba5142f090171c2d8ccd143e`
- submission SHA256: `7f5d9156a732531148f15680cd0583a4df4418c440b4d3e93ad2fef9336da8ea`

## Hidden submit failure

提出時の hidden rerun で `Notebook Threw Exception` が発生した。原因は inference v3 が exp145 の public raw-test learned-likelihood feature に依存していたこと。hidden test では ID が一致しないため current-test feature generation に fallback するが、exp162 config に exp145 generator contract と exp111 artifact paths が不足しており、`generator.candidates must not be empty` で落ちる。

ローカルでは修正済み。exp162 config に exp111 paths と generator 設定を追加し、inference は `learned_feature_path=None` にして public / hidden とも current-test feature generation を通す。`exp145-inference` は kernel source から外した。

ただし Kaggle への修正版 push は未完了。`exp161` 2本と `exp163` 3本が `RUNNING` で CPU session 上限5本を使い切っており、`kaggle kernels push` が `Maximum batch CPU session count of 5 reached` で拒否されている。修正版 inference kernel を push して public rerun が完了するまで、exp162 は再提出しない。

## Inference v4

修正版 inference v4 は Kaggle 上で `KernelWorkerStatus.COMPLETE`。`exp145-inference` を kernel source から外し、public / hidden とも current-test learned-likelihood features を生成する経路で実行した。

- kernel: `kentookumura/exp162-ll-rank-slot-exp148-infer` v4
- status: `KernelWorkerStatus.COMPLETE`
- raw-test learned likelihood source kind: `target_free_current_test_generated_learned_likelihood_ml_features`
- learned feature rows / wells / columns: 14151 / 3 / 51
- learned feature decompressed SHA256: `27efc7c7ef776fc21a9792c8e1a587d4f9fc99a0b2e7945cd8d47d165c658fbb`
- loaded boosters: 15
- feature count: 375
- test rows / submission rows / predicted rows: 14151 / 14151 / 14151
- fallback rows: 0
- prediction range: 11590.8623046875 - 12240.15234375
- prediction mean / std: 11905.147261360173 / 278.8603867428979
- prediction SHA256: `16ad86b3d400c3aa0bfd67e86e6340eda0d8293d919011694df81d6499b0b7da`
- submission SHA256: `75c7374ae07314e996d69968cee3743f4119e6d6229ac5339195ee0107777571`
- elapsed seconds: 155.072

Submit-check は PASS。`submission.csv` は sample submission と header / row count が一致し、重複 ID、empty / NaN / Inf-like values は検出されなかった。

## Submission

- ref: `54247043`
- date: `2026-07-02 00:02:06.833000`
- status: `SubmissionStatus.COMPLETE`
- Public LB: `8.100`
- Private LB: `-`
- submission SHA256: `75c7374ae07314e996d69968cee3743f4119e6d6229ac5339195ee0107777571`
- comparison: exp148 Public LB `7.960` から +0.140、exp160 Public LB `8.061` から +0.039 悪化

CV では単体 `lgb1` / `lgb2` が exp148 `lgb_mean` を上回ったが、Public LB は悪化した。exp162 は採用せず、exp148 を ML route anchor として維持する。
