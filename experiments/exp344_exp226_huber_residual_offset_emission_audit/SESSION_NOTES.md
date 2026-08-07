# exp344 セッションノート

## 2026-07-22 設計確定

- 目的: Student-tの結果を見た後の無制限な手法選択を避け、Huberを使う条件を事前登録する。
- 依存: exp342の極端残差改善と全体flattening失敗が同時に確認された場合だけ解禁する。
- Stage 0規模予約: Huber scientific readout 1、Gaussian保存済みcontrol、circular control、HMM run 0。
- Stage 1規模予約: 全gate通過と再承認時のみvariant 1、fold 5、773 well HMM run。
- long-tail guard: pooled、4/5 fold、stress、shuffle、極端残差bucketを固定した。
- 禁止: 結果を見たdelta調整、cap併用、Student-tとの同時比較sweep。

## 未実施

コード実装、notebook実行、Kaggle push、成果物生成は行っていない。

## 2026-07-23 依存判定

- exp342 Stage 0はextreme-residual top3/regretを改善したが、
  pooled gainとstress非劣化をFAILした。
- exp342のStudent-t likelihood top1 marginはGaussianより低下せず、
  flattening signalはfalse。
- 事前指定した「extreme改善 + pooled FAIL + flattening」のAND patternは不成立。
- 結果を見たpost-hoc Huber選択を避け、exp344は未実装・未実行で閉じる。
- Stage 0/1、Kaggle、inference、submission、delta/cap/scale救済は全て0。
