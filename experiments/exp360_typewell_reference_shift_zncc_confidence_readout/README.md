# exp360_typewell_reference_shift_zncc_confidence_readout

## 状態

- ルート: `ensemble`
- 状態: Stage 0 gate FAIL、branch closed
- CV: primary pooled bad10 AUC `0.505164`
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-23
- 親実験: `exp340_exp226_depth_alias_block_confidence_readout_on_exp264`
- 比較対象: `exp280_exp226_shift_likelihood_separability_readout`
- error readout: `exp264_exp263_candidate_confidence_dual_selector`

## 仮説

exp280 の raw Gaussian likelihood は absolute GR residual を測るため、振幅・offset の差と
shape mismatch が混ざる。raw finite horizontal GR と `GR_typewell(TVT_geop + δ)` の
ZNCCなら、通常 matchingの `δ=0` に対する非ゼロshiftの優位を、scale/offsetに依存しにくい
target-free confidenceとして測れる可能性がある。

この実験は「14 ftずらせば正解」と仮定するものではない。全773 OOF wellsで `δ=0` を残し、
固定13-shift surfaceを測り、shiftが必要か不明な状態そのものをconfidenceへ変換する。

## 変更点

- exp226 path、exp280の512-row blockと13-shift bank、exp264 late error readoutを固定する。
- 単一の科学的変更は、raw absolute Gaussian scoreからraw-finite ZNCCへの置換。
- shiftはhorizontal row/MDを動かさず、Type Well参照を
  `GR_typewell(TVT_geop + δ)` として動かす。
- primaryは `best_nonzero_minus_zero_zncc` だけとする。
- exp280保存scoreをmatched control、stable SHA256 shift-label permutationをnegative controlにする。
- prediction、candidate、selector、モデル、推論、提出は変更しない。

## 検証方針

- Fold: exp226 group-safe 5 foldsを固定。
- Group: `well_id`。
- Score unit: unknown suffixの512-row fixed non-overlapping block。
- Shift bank: `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`。
- ZNCC: raw finite pair 32以上、observed/expected std双方 `>1e-6`。
- Coverage: supported blocks `>=0.98`、全773 wellsにsupported blockが1つ以上。
- Leakage Check: score、control、feature、fold quantile、manifest、SHAをfreezeするまで
  true TVT、exp264 error、bad10 labelを開かない。
- Scientific gate: primaryのRMSE quartile lift、4/5 folds、bad10 AUC `>=0.60`、
  1000+/hidden-like、raw Gaussian比 `+0.02 AUC`、permutation比 `+0.02 AUC`をAND判定する。
- Sentinel: 000d7d20、8cc21f01、e5ff9fd2、2ddad940は説明表示だけで、設計や合否を変えない。

## 実行入口

- 学習 notebook: `exp360_typewell_reference_shift_zncc_confidence_readout_train.ipynb`
- 推論 notebook: `exp360_typewell_reference_shift_zncc_confidence_readout_inference.ipynb`
- Jupytext percent形式のcompact self-contained sourceから正規Notebookを生成済み。
- `implementation_approved: true`、`run_stage_0: true`。Kaggle private CPU
  canonical version 2で実行済み。
- inference / submission は設計上も無効。

## 結果

| メトリック | 値 |
| --- | --- |
| primary pooled bad10 AUC | 0.505164 |
| Q4−Q1 mean block RMSE | +0.107479 ft |
| raw Gaussian比AUC | −0.044785 |
| permutation比AUC | +0.016644 |
| supported blocks / wells | 7,700 / 772 |
| gate | FAIL |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 7,787 expected blocksと5 foldsを維持し、truth access 0のままfeature/SHAをfreezeした。
- 13件の生成物SHAはmanifestと一致し、raw Gaussian / permutationと同じblockで比較できた。
- 0-booster、親control再学習なし、125.4秒で仮説を反証できた。

### 悪かった点

- primary AUCは0.505164でgate 0.60に届かず、raw Gaussianより0.044785低かった。
- Q4−Q1 meanは+0.107479 ftだけで、1000+では−0.169027 ftへ反転した。
- `896d15b9`にsupported blockがなく、全773 wells support条件もFAILした。

### リスク / 注意

- 4 sentinel wellsに合わせたdense grid、閾値、pair条件の変更は禁止。
- supporting familyでprimary failureを救済しない。
- primary scientific gateもFAILしたため、add-only利用や予測へのshift適用へ進まない。

## 次

- ZNCC confidence branchを閉じる。
- threshold/family/shift grid/pair条件による救済、再実行、inference、submissionを行わない。
- 同familyの救済backlogを追加せず、既存の上位候補を優先する。

## 表記

用語は `backlog/KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
