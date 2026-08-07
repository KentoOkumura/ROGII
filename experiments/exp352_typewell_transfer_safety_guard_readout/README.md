# exp352_typewell_transfer_safety_guard_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU Stage 0完了、worst-well gate FAIL、branch closed
- CV: Stage 0 gate 7/8 PASS、総合FAIL
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-23
- 科学的参照: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- 履歴参照: `exp313_typewell_group_unseen_transfer_guard`

## 仮説

`exp311`のType Well群priorは平均的なGR再構成を改善した一方、worst-wellを大きく悪化させた。
peer/supportとfallbackだけで構成したtarget-free guardが、same-group gainを保持しつつ
未知群・purged wellへのnegative transferを防げるなら、群情報には補正ではなく
「利用可否診断」として独立した価値がある。

## 変更点

- 旧`exp313`を再開せず、`exp311/312`のpromotion PASSを要求しない新番号へ切り出した。
- exp311保存値を変えず、availability、support、fallbackだけを監査する。
- PASSしても旧exp314--320を自動解禁しない。

## 検証方針

- Fold: exp311保存済みouter 5 folds。
- Group: native Type Well overlap group。
- Audit surfaces: same-group holdout、leave-one-group-out、spatial+typewell purge。
- Leakage Check: availability/fallback manifestをouter-valid truth/error結合前にSHA固定する。
- Gate: coverage 0.90、same-group gain 0.05 horizontal GR API、4/5 folds、
  2 purged面negative transfer 0、identity parity `1e-10`、
  worst `+0.25 horizontal GR API`をAND条件とする。
- exp311はTVT予測を生成していないため、事前登録した数値は変えず、誤っていた`ft`表記を
  保存scoreと同じhorizontal GR APIへ訂正した。

## 実行入口

- 正規trainのJupytext参照元:
  `exp352_typewell_transfer_safety_guard_readout_compact_selfcontained_train.py/.ipynb`
- fail-closed inference:
  `exp352_typewell_transfer_safety_guard_readout_compact_selfcontained_inference.py/.ipynb`
- compact trainを正規train notebookへ採用してKaggle CPU version 1を実行した。
- inference notebookはplaceholder/fail-closedのまま。
- notebook 実行: Kaggle kernel run を正とする。ローカル実行は `--allow-local` を付けた smoke debug のみに限定する。

## 結果

| メトリック | 値 |
| --- | --- |
| exact coverage | 0.972833 |
| same-group gain | +0.381540 GR API / 5 of 5 folds |
| leave-group-out negative transfer | -0.164862 GR API |
| spatial+typewell-purged negative transfer | -0.496752 GR API |
| worst-well regression | +12.914716 GR API（FAIL） |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- exp311の平均gainとtail failureを、補正値の再調整なしに安価に分離する設計である。
- pooled gain、coverage、identity parity、未知群/purged non-regressionは固定gateを通過した。

### 悪かった点

- exact groupを許可した`d07aed8f`が+12.914716 GR API悪化し、worst safetyを再現できなかった。

### リスク / 注意

- 同じreadoutでpeer/support閾値やfallback順を変更しない。
- Type Well群coverageのhidden test差を3 audit surfacesで保守的に扱う。

## 次

- 同じreadoutでthreshold/fallbackを救済調整せずbranchを閉じる。
- exp311/312のFAILと旧exp314--320の閉鎖を維持する。
- 既存exp353はsoft quality featureの独立preflightとしてのみ扱い、自動昇格しない。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
