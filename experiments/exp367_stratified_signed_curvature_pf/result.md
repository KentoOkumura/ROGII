# exp367_stratified_signed_curvature_pf 結果

## 仮説

各符号modeを層別維持すれば、GRが支持するsigned-curvatureをPFが失わず追跡できる。

## 設定

- 親: `exp072_exp063_full_replay_feature_cache`
- route: `pf_beam`
- 実行: Kaggle private CPU version 1
- kernel: `kentookumura/exp367-stratified-signed-curvature-pf-train`
- id_no: `128528103`
- runtime: `267.914282461 sec`
- Stage 0: PFを回さない固定3 signed pathのtruth-late GR識別監査
- reporting folds: 5
- PF seed-well runs / control replay: `0 / 0`
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- seed: stable SHA256 per well / seed。Stage 0の固定3軌道はRNGなし

## 結果

decisionは`stage_0_failed_close_without_rescue`。technical gateはPASSしたが、
scientific gateはFAILし、Stage 1は不適格となった。

| 評価面 | 結果 | 条件 | 判定 |
|---|---:|---:|---|
| overall top1 | 0.469591 | 0.40以上 | PASS |
| top1 gain vs zero-first | +0.461081 | 参考 | - |
| MRR | 0.687550 | 参考 | - |
| MRR gain vs zero-first | +0.276771 | 0.01以上 | PASS |
| real - circular top1 | +0.005576 | 0.03以上 | FAIL |
| passing folds | 2/5 | 4/5以上 | FAIL |
| selected path RMSE gain vs zero | +0.829601 ft | 参考 | - |
| 1000+ RMSE gain vs zero | +0.911161 ft | 正方向 | PASS |
| hidden-like spatial RMSE gain vs zero | +0.306996 ft | 正方向 | PASS |
| hidden-like typewell-purged RMSE gain vs zero | +0.447147 ft | 正方向 | PASS |

- inputは773 wells、完全512-row blockを作れた772 wellsを採点した。
- block数は13,631、block overlapを含む評価行数は6,979,072。
- fold別scientific gateはfold 3/4だけがPASSし、fold 0/1/2はFAILした。
- truth-before-freeze 0、hidden-like role-before-freeze 0、candidate/score SHA readback、
  ID/key一意性、512-row block、identity、PF run 0、Stage 1無効を含むtechnical checkは
  すべてPASSした。

## 解釈

固定signed path自体はzero-first基準より選べており、全stress scopeのRMSE方向も正だった。
一方、top1はcircular GR controlでも0.464016で、real GRの上積みは0.005576にすぎない。
foldでも正方向が2/5に留まったため、識別力は局所的なGR整合ではなくblock構造や順位規則でも
再現できる可能性が高い。98,944 seed-well runsを要するStage 1 PFへ進む根拠として不十分である。

事前契約どおり、circular差やfold条件の事後緩和、quota / curvature / transitionの探索、
Stage 1 PF、inference、submissionによる救済は行わず、この分岐を閉じる。

## 再現性

- scientific contract SHA256:
  `c5893932790ac5194dc62ff84e5dda103cff2f3270c1bc96ea026f3c159ccb44`
- candidate paths decompressed SHA256:
  `7d3301c309494845ffcd7f24fa1863f5c551d455ed9282110ba2a6482a227386`
- block GR scores decompressed SHA256:
  `a7b5cc9d538e1eb040e42086477a06157d42cbec0b61aac79c1f28b221258b0a`
- freeze manifest SHA256:
  `9e3da3d462fdf77cf5177ce37793566938d50be34f610dfaf9397f235a811f3c`
- gate report SHA256:
  `338d78383fb01faef993de57d1b235f81eea02b92c79f7093ac517f3a8bfec4d`
- Kaggle log SHA256:
  `a3272196ebc84a409a88cb04d38f6a3a448a0a421c86be32158c16f875cfcf53`
- CV / Public LB / Private LB: なし
- submission SHA: 生成なし

全SHAは`metrics.json`と`config.yaml`の`results.content_sha256`を正とする。

## 次

exp367は完了・閉鎖。Stage 1 PF、inference、submissionは実装・実行しない。
同じ仮説のparameter rescue backlogも追加しない。
