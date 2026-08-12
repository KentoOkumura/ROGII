# 要件

## 依頼

exp226の「学習坑井ごとにTVTの地層ドリフトを16区間の傾きへ分解し、近傍坑井から対象位置へ補間する」考え方へtrain-only地層面を組み込み、まず信号の識別可能性を調べる。

2026-07-24のユーザー指示「exp377を実装してください」を、Stage 0 integrityと、Stage 0 PASS時だけ動くStage 1 identifiabilityのimplementation-only承認として受領した。正規Notebook上書き、Kaggle package/push/run、current-test生成、inference、submissionは今回の承認範囲に含めない。

2026-07-24の追加指示「実行してください」により、正規train Notebook採用と
1 diagnostic / 6 reporting surfaces / 5 folds / model・HMM・PF・booster・
parent-control再実行各0のKaggle CPU v1を承認済みとした。inference、submission、
current-test生成、Stage 0不合格時の救済再実行は承認範囲に含めない。

2026-07-24の追加指示「1を実行してください」により、K16、近傍50、bandwidth
500 ft、ridge 1、formation-relative式、primary、Stage 1 gateを一切変えず、
effective-donor checkだけを共通kernelのreport-only診断へ変更するKaggle CPU v2を
1回承認済みとした。truth late join後のdirect対formation-relative Stage 1比較までを
実行範囲とし、kernel parameter変更、追加variant、inference、submissionは含めない。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- outer-validおよびtestの生のtrain-only地層列は読まない。
- donor側の地層列と正解TVTはouter-train内に限定する。
- 6地層を結果確認後に選ばず、robust medianをprimaryとして事前固定する。
- HMM、PF、LightGBM、親control再学習は行わない。
- compact self-contained候補をJupytext起点で実装し、既存の正規Notebookは上書きしない。
- `h512`は未知suffix先頭512行、`long401`は未知suffix長401行以上の坑井とする。`clean273`は行集合ではなくML特徴allowlistなので、本readoutでは架空の坑井集合を作らずpooled契約の別名として明記する。

## 受け入れ基準

- 3,783,989行、773坑井、12,368区間が一意に復元され、対象側地層列read countが0である。
- 地層面予測のfallback率が5%以下、primary coverageが98%以上、effective donor数p05が10以上である。
- donor rate RMSEがexp226比5%以上改善し、累積path RMSEが0.50 ft以上改善する。
- 5 fold中4 fold以上で改善し、H512/long401/clean273の悪化が各0.02 ft以下、p95坑井悪化が0、worst坑井悪化が0.25 ft以下である。
- deterministic anchorとして、fold manifest、地層面、6相対勾配、median系列のcontent SHAとkernel versionを記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
