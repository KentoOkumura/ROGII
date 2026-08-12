# exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm

## 状態

- ルート: `ensemble`
- 状態: Stage 0 Kaggle CPU version 1 完了 / hard gate FAIL / branch closed
- CV: なし
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-19
- 親実験: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`

## 仮説

`exp223` の self-GR donor window にある raw GR 欠損だけを、known `TVT_input` で Type Well から復元し、observed known GR にwell単位で頑健 affine 校正した値で補完する。observed GR、元のmissing mask、anchor eligibility、target receiver、base Type Well HMMを固定すれば、欠損線形補間が作る偽の局所形状だけを減らせる可能性がある。

## 変更点

- finite raw GR は変更しない。
- known prefix の raw missing donor cellだけを `a * TypeWellGR(TVT_input) + b` で補完する。
- Type Well範囲外またはaffine fit無効時は、`exp223` の線形補間へ戻す。
- raw missing mask、self-GR anchor/window eligibility、target receiver、base HMM emissionを固定する。
- self-GRは `alpha=0.07 / clip=1.0 / boost_only` の1本だけとし、gridを作らない。
- Type Well GR全面置換、target区間復元、state-known emission、inference、submissionは行わない。

## 検証方針

- Stage 0: known prefix の自然欠損run長から作る deterministic pseudo-missing blockで、既存線形補間とType Well gap-fillを比較する。pooled RMSE 5%以上、ZNCC `+0.02`、4/5 reporting folds、by-well p95非悪化を必須にする。
- Stage 1: Stage 0全PASSと別承認後のみ、固定exp223 HMMを1 variant / 773 well-runsで比較する。exp223 11.349950650から0.10 ft以上、4/5 folds、1000+/hidden-like、worst-well、missing-rate scopeの全guardを必須にする。
- Group: `well`。
- reporting fold: stable SHA256 well-hash 5分割。学習foldは0。
- leakage check: pseudo-mask rowのaffine fit除外、truth-late-join、raw-mask parity、target-side Type Well fill 0。

## 実行規模

| 段階 | variant | HMM well-runs | LightGBM config | trained fold | booster | GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 0 | 1 audit | 0 | 0 | 0 | 0 | 0 |
| Stage 1 | 1 HMM | 773 | 0 | 0 | 0 | 0 |

親/controlは保存済み `exp223` を使い、再実行しない。Stage 1の想定CPU時間は約5-6時間。

## 実行入口

Stage 0 は次の別名 compact self-contained sourceを正として実装し、2026-07-19のユーザー実行承認により正規train notebookへ採用する。

- `exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm_compact_selfcontained_train.py`
- `exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm_compact_selfcontained_train.ipynb`
- `exp294_calibrated_typewell_gapfill_known_prefix_selfgr_hmm_train.ipynb`（採用済み正規notebook）

承認範囲どおりKaggle CPU Stage 0 audit 1件、LightGBM config / trained fold / booster `0 / 0 / 0`、HMM/PF well-run 0、親control再学習なしで完了した。実行後は`execution.run_stage0: false`へ戻し、Stage 1 / inference / submission を閉じた。

## 結果

Kaggle CPU version 1（id_no `127890033`）は773 wells / 2,319 blocks / 3,865 held-out rowsを160.32秒で完了した。control RMSE 8.138531に対しType Well gap-fillは12.842186（`+4.703655 ft`、相対改善`-57.7949%`）、改善fold 0/5、by-well p95 delta `+15.494311 ft`、157 wells改善 / 610悪化 / 6同値でperformance hard gateを全FAILした。

observed/raw-mask parity、pseudo-mask fit除外、target fill 0、finite coverage 1.0、truth-late-joinなどtechnical gateはPASSし、取得した9生成物のbyte数・SHAはmanifestと全一致した。自然欠損run長が全foldで`1/1/3`行のためZNCCは未定義だが、RMSEとp95だけで棄却は確定する。

## 所見

### 良かった点

- 設計段階で変更対象をknown-prefix donorのraw missing cellだけに絞り、observed GR、raw mask、anchor eligibility、target receiver、base HMMを不変条件にできた。
- Stage 0の信号復元監査をStage 1の約5-6時間HMMより先に置き、失敗時の停止条件を固定できた。

### 悪かった点

- Type Well gap-fillはpooled・全fold・well tailのすべてで線形補間より大幅に悪化した。
- well単位robust affineは767/773 wellsで成立したが、局所欠損GRの復元には不十分だった。

### リスク / 注意

- pseudo-mask rowのfit混入、raw maskを補完後に再計算する実装、target-side Type Well fillはリークまたは原因分離破壊になるためhard failとする。
- Stage 0/1のFAIL後にaffine、window、alpha、thresholdの救済gridを追加しない。

## 判定境界

Stage 0 FAILによりStage 1へ進まず、救済gridなしでbranchを閉じた。raw-test inference / submissionは行わない。

## 設計の正

`docs/legacy/steering/20260719-exp294-calibrated-typewell-gapfill-known-prefix-selfgr-hmm/` を正とする。
