# 要件

## 依頼

train-only地層列と正解TVTの関係を、局所勾配ではなく「坑井が地層面を横切る順序と接触位置」として物理モデルへ利用する。まず0-HMMで接触予測可能性を監査し、合格した場合だけ順序制約付きsemi-Markov HMMへ進む。2026-07-24の実装指示ではStage 0だけをcompact self-contained notebook候補として実装し、Stage 1は別承認のままにする。

## 制約

- Route: `pf_beam`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 地層順を `ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA` に固定する。
- outer-valid/testの生Formation列を読まず、outer-trainから地層面を予測する。
- target側補正はknown prefixから推定する単一additive offsetだけとする。
- Stage 0不合格ならHMMを実装しない。
- Stage 1の773坑井HMMは別途ユーザー承認を要する。
- 既存の正規notebook scaffoldは上書きせず、`*_compact_selfcontained_{train,inference}.py` / `.ipynb`を候補として作る。
- Kaggle package、push、run、正規notebook採用、推論、提出は今回の実装範囲に含めない。

## 受け入れ基準

- Stage 0でeligible well率25%以上、contact event 1,000件以上を確保する。
- crossing MD MAE 128 ft以下、p90 512 ft以下、contact-TVT RMSE 15 ft以下、正しい接触順率95%以上である。
- constant-surface基準より0.10 ft以上改善し、5 fold中4 fold以上で正である。
- Stage 1はexp209より0.05 ft以上改善、5 fold中4 fold以上正、scope悪化0.02 ft以下、p95悪化0、worst悪化0.25 ft以下である。
- 固定blendでも0.02 ft以上改善する。
- surface、crossing、contact target、duration prior、HMM predictionのSHAを記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。
