# 要件

## 依頼

`exp029_public_sel15_pf_oof_feature_generation` の train-side OOF-like artifact を使い、公開 sel15 PF/Beam 候補を行または well 単位で選択できるか監査する。

## 制約

- Route: `pf_beam`
- 入力は 見えない test で使える に生成済みの `public_sel15_pf_oof_features.csv.gz` を使う。
- `target_tvt` は候補生成には使わず、selector 監査と error 集計にのみ使う。
- same-OOF の best / oracle は診断値として扱い、採用判断には使わない。
- `exp026_oof` は exp029 artifact で未接続なので、今回の候補セットには含めない。接続できたら後続で再監査する。

## 受け入れ基準

- 候補別 RMSE、distance bucket 別 RMSE、well-level win/loss を artifact と metrics に保存する。
- fixed candidate、rule selector、conservative blend / fallback 候補を比較する。
- leave-one-original-fold-out selection と stable well-hash holdout selection の両方を出力する。
- 両 holdout で raw anchor 候補と public PF 単体を上回る場合だけ inference 化を支持する。
- 改善が不安定な場合は hard selection を不採用にし、保守的 blend / fallback rule の次候補を記録する。
