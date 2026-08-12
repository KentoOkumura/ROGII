# 要件

## 依頼

exp209 exact HMMの遷移モデルへ、base prefix-rateと6種類のformation-relative K16 rateをmodeとして組み込み、地層ごとの参照度合いをHMM内で推定する。今回は設計のみ確定する。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- exp377の識別可能性とexp378の候補新規性が合格するまで実装しない。
- mode切替はK16境界だけに限定し、区間内では固定する。
- 遷移確率を事前固定し、rescue gridを行わない。
- Stage 0は16坑井だけ。773坑井Stage 1は別途ユーザー承認を要する。
- 親controlを再実行しない。

## 受け入れ基準

- base modeだけの出力がexp209親出力とRMSE差1e-8以下で一致する。
- posterior正規化誤差1e-8以下、予測full runtime 30,600秒以下、peak RSS 25 GB以下である。
- posterior base mass平均が0.05〜0.95に入り、modeが退化しない。
- Stage 1はexp209直接HMMより0.10 ft以上改善、5 fold中4 fold以上正、各scope悪化0.02 ft以下、p95悪化0、worst悪化0.25 ft以下である。
- 固定likelihood-PF blendでも0.05 ft以上改善する。
- mode順、K16境界、遷移行列、入力候補SHA、audit well listを記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
