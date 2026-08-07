# 要件

## 依頼

exp490を適用すべきwellと、保存済みexp357予測へ戻すべきwellを、未知suffixの
正解を見る前に見極められるか調査する。

## 制約

- Route: `ensemble`。ML selectorとPF/HMM候補の両方が最終予測選択に本質的に寄与する。
- 保存済みexp490 full OOFとexp498 target-free well特徴だけを使い、PF/HMM、exp357、
  exp226、controlを再実行・再学習しない。
- outer 5-foldを完全なheld-out単位とし、モデル族の選択も各outer-train内のinner 4-fold
  だけで行う。outer-validの正解、誤差、fold別成績でthresholdやモデルを救済しない。
- selector入力に`fold`、真値、candidate/parent error、truth由来roleを含めない。
- 一つのwellにはexp490全体かexp357全体のどちらか一方だけを適用する。
- inference、submission、LB auditは本実験の範囲外とし、train-side gate通過後も別途承認を要する。
- 再現性は`docs/06_reproducibility.md`に従い、入力、特徴、OOF selector score、
  model manifestのSHAを保存する。

## 受け入れ基準

- 3,783,989行、773 wells、5 foldsをSHA-pinned入力から過不足なく再構成できる。
- target-free 32特徴を正解・誤差・foldより先にfreezeし、feature contract/content SHAを記録する。
- pooled/fold別AUC、Spearman、適用率、beneficial precision、policy RMSE、by-well tailを出力する。
- report-only oracle、never-exp490、always-exp490、cross-fitted policyを同じwell/row重みで比較する。
- `predictability_supported`はpooled AUC 0.60以上、AUC 0.55以上が4/5 folds以上、
  Spearman正方向が4/5 folds以上をすべて満たす。
- `safe_router_supported`はalways-exp490比0.05 ft以上改善、4/5 folds以上で非悪化、
  適用率20–95%、selected-minus-exp357 by-well RMSE p95が0.25 ft以下、worstが5 ft以下をすべて満たす。
- 技術検査、predictability、safe routerがすべて通った場合だけ後続inference設計を許可する。
- gzip入力はraw SHAに加えてdecompressed content SHAを主証拠として記録する。

