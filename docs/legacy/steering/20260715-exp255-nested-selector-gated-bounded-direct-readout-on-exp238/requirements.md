# 要件

## 依頼

exp238のselector 1位候補をLightGBMの特徴として渡すだけでなく、誤選択を抑えるgateと移動量上限を通して最終予測へ明示的に反映する。

## 制約

- Routeは`ensemble`。exp238 ML予測とPF/Beam/HMM/exp226候補が最終予測へ本質的に寄与する。
- exp238 outer-valid selector score、exp238 final OOF、既存OOF候補だけを固定入力とする。
- selector、LightGBM、candidate、parent/controlの新規学習は0。
- selectorの候補選択、gate、alpha、clipにouter-valid truth/error/oracleを使わない。
- conservative/balanced/assertiveの3 profileを実行前に固定し、target-selected grid searchを行わない。
- hard top1は診断だけとし、提出候補へ昇格させない。
- Kaggle CPU / internet off。submission生成・competition submitは行わない。
- gzip生成物はdecompressed content SHAを主証拠にする。

## 受け入れ基準

- 各rowが同じouter foldのrole=`valid` selector scoreだけを使う。
- 3,783,989行 / 773 wellsを重複・欠損なく被覆する。
- correctionはselector top1方向だけで、profile別最大移動量4/7.5/12 ftを超えない。
- global改善、near/1000+/hidden-like非悪化、3/5 folds改善、worst-well +0.25以下を全て満たすprofileだけを採用可能とする。
- 全guard不通過ならinference/submitへ進めない。
- metrics/by-well/gate/candidate distribution/selected OOF/input manifest/summaryとSHAを保存する。

