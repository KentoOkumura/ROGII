# 要件

## 依頼

最終提出2枠のうちprivate一般化を優先する第1枠として、保存済みexp413 current-test予測と、
Public-LB固有overlayを除去してfold-safeに再構成したexp497 strict public-core予測を固定係数で
混合する。exp413単独ではなくpublic trajectoryの相補性を残すが、exp497の科学的promotion
gate失敗を覆して新しいanchorに昇格させる実験とは扱わない。

## 制約

- Route: `ensemble`
- 親anchor: `exp413_scale5_likpf_full_replacement_on_exp335`
- auxiliary source: `exp497_strict_public_core_fold_safe_ensemble_on_exp413`
- 最終式を次へ固定する。

```text
public_core_weight = 0.13716473330712417
prediction = (1 - public_core_weight) * exp413 + public_core_weight * strict_public_core
```

- 係数はexp497の5 meta-fold係数の中央値であり、再fit、grid、Public LBによる変更を禁止する。
- exp413、exp497 component、selector、PF/Beam、modelを再学習・再生成しない。
- Gold visible-prefix overlay、guarded contact override、同一well lookup、public well ID規則を使わない。
- row/well/confidence router、conditional blend、intercept、clip、SG、warmup、再anchorを最終blend後に追加しない。
- exp497 Stage I version 4のstrict public-core model/prediction生成物をSHA固定入力とする。
- exp497のpromotion gate `FAIL`は履歴として維持し、本実験を科学的anchor更新とは表記しない。
- 2026-08-04の実装承認範囲は候補source/notebook、契約test、記録まで。正規notebook採用、
  Kaggle package/run、output取得、提出は引き続き各別承認とする。
- 再現性: `docs/06_reproducibility.md` に従い、入力predictionのlogical SHA、blend prediction SHA、submission SHA、kernel versionを記録する。

## 受け入れ基準

- 2 componentを`id`でone-to-one結合し、sample ID/order、row count、nonempty well、NaN/Inf、重複をfail-closedで検証する。
- exp413 current-test predictionは既知SHA契約と一致し、exp497 strict public-coreはStage I完了後に固定されたprediction/content SHAと一致する。
- float64で固定式を1回だけ評価し、componentごとのweightが全行一定、weight合計が厳密に1である。
- 生成物にexp413、strict public-core、final blendの3列、component差分要約、入力/出力SHA manifestを含める。
- 実装後のtechnical gateを全通過した場合だけ「最終提出第1枠候補」と表記する。CV promotion PASSとは表記しない。
- deterministic anchorとして扱う場合は、prediction SHA、submission SHA、Kaggle kernel versionのrerun一致を記録する。
- gzip生成物を比較する場合はdecompressed content SHAを主証拠として記録する。
