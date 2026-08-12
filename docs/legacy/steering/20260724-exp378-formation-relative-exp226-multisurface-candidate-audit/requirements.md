# 要件

## 依頼

exp377で識別可能と判定されたformation-relative K16勾配をexp226の物理パスへ戻し、6地層単独と固定medianの7候補を作る。直接精度と既存候補集合に対する新規性を監査する。今回は設計のみ確定する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp377のintegrityとidentifiabilityが合格するまで実装しない。
- exp226のrate prior以外の処理を固定する。
- 7候補は別系列で保存し、CV後のbest-of-7をprimaryにしない。
- HMM、PF、ML学習、親control再実行は行わない。

## 受け入れ基準

- 7候補すべてが3,783,989行・773坑井を満たし、対象側地層列read countが0である。
- Primary medianがexp226より0.05 ft以上改善、5 fold中4 fold以上改善し、各scope悪化0.02 ft以下、p95坑井悪化0、worst坑井悪化0.25 ft以下である。
- exp226固定12候補へ7候補を全追加したH512 oracle gainが0.15 ft以上、whole gainが0.10 ft以上、全5 fold正、unique-best行率10%以上である。
- exp226候補との最大絶対相関が0.9995以下である。
- exp377入力SHA、候補順、7系列のdecompressed content SHAを記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
