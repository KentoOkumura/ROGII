# exp323 セッションノート

## 目的

exp226 geometryを予測blendではなく、exact HMMの時間変化するdip-rate prior平均として移植する。

## 現在の状態

- 2026-07-21: steering、実験scaffold、2-stage設計を確定。
- Route: `pf_beam`
- 状態: terminal closed / 未実装 / 未実行
- 実行量: Stage 0は1 diagnostic・HMM 0、Stage 1最大1 variant・773 HMM runs、0 model / 0 booster。
- control再実行、GPU、inference、submission: すべて0または無効。

## 設計判断

- exp279のabsolute unaryとexp281の固定exp226 shapeを再試行しない。
- exp273の対象well prefix 2D planeではなく、exp226のouter-fold donor K16 fieldを使う。
- 絶対rateは親HMMの初期rateに残し、`r_geo,t-r_geo,first`だけを遷移平均へ入れる。
- Stage 0のtarget-free scheduleとSHAを凍結した後だけsuffix truthを結合する。

## 再現性メモ

- RNGなし、outer fold / well / segment順を固定する。
- exp226 geometry field、rate schedule、input manifest、predictionのdecompressed content SHAを記録する。
- deterministic submission anchorではない。Kaggle kernel、prediction SHAは未生成。

## 2026-07-22 閉鎖

- exp307 promotion FAILによりexp308/309固定lineageが成立しないため、未実装・未実行のまま閉鎖した。
- exp323自体のreparent、実装、Kaggle実行は今後行わない。
- exp338が全promotion gateをPASSした場合だけ、exp338を親に新exp323相当を新番号・別steering・別承認で作る。

## 次

なし。新exp323相当の資格判定はexp338の記録を正とする。
