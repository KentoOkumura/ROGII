# exp350_exp345_bidirectional_gr_affine_smoother

## 状態

- ルート: `pf_beam`
- 状態: `stage_0_failed_closed`
- CV / Public LB / Private LB: 14.367548 / - / -
- 作成日: 2026-07-22
- 科学的親: `exp345_exp209_time_varying_gr_affine_calibration_hmm`
- root parent: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 科学実装 / 正規train Notebook採用: 完了
- Kaggle Stage 0: version 1完了、technical PASS / scientific FAIL
- Stage 1 / inference / submission: 不適格・未実施

## 仮説

exp345は現在行までのraw GRだけでaffine stateを更新し、pooled RMSEを`0.169505 ft`改善したが、400/773 wellsが悪化しworstは`+9.354827 ft`だった。推論時に利用可能な井戸全体のraw GRで同じstateを前後方向に平滑化すれば、平均gainを保ちながら局所的な誤calibrationとworst tailを抑えられる可能性がある。

## 単一変更

exp345のcausal EKFで得る`[intercept_b, log_scale_a]`へ、固定区間extended RTS backward passを1回追加する。

```text
exp345: visible prefix + 過去～現在GR → causal affine schedule → HMM
exp350: 同じforward記録 + 井戸末尾までのGR → bidirectional RTS schedule → HMM
```

exp345のinitial fit、process noise、observation、slope boundと、exp209のsigma、missing weight、transition、state grid、prior、posterior-mean decoderは変更しない。exp345は閉鎖のまま維持し、保存済み成果物をcontrolとしてだけ使う。

## 推論契約

full-well raw GRは推論入力として利用可能なので使用を許可する。last-640 `TVT_input`、true TVT、error、formation、fold、hidden-like roleはprediction freeze前に使用しない。本candidateはoffline full-well専用であり、streaming/causal methodとは扱わない。

## 検証方針

- Stage 0: exp345と同じlast-640 mask、773 wells / 494,720 score rows。
- control: exp345保存masked exp209 parentとcausal candidate。control HMM再実行0。
- candidate: forward filter 773 + smoother 773 + new exact-HMM 773 runs。
- fold: exp226固定5 folds、Group=`well_id`。fold/hidden-likeはprediction SHA freeze後にlate joinする。
- Technical: input SHA、forward parity、terminal/covariance、finite、coverage、runtime`<=8.5 h`。
- Scientific: parent比`>=0.05 ft`、causal比`>=0.02 ft`、両方4/5 folds、hidden-like 2面、by-well median/p95、worst`<=+0.25 ft`、boundaryをAND gateにする。
- GR reconstruction NLLは未来GRを使うため診断専用とし、promotionには使わない。
- 全gate PASS後もStage 1は別承認。FAIL時はparameter rescueなしで閉じる。

## 実行量

| 項目 | 現在 | Stage 0実装・実行が別承認された場合 |
| --- | ---: | ---: |
| scientific variant | 1 | 1 |
| new HMM well-runs | 773 | 773 |
| parent / causal control HMM rerun | 0 / 0 | 0 / 0 |
| LightGBM config / trained fold / booster | 0 / 0 / 0 | 0 / 0 / 0 |
| PF / Beam / GPU | 0 / 0 / 0 | 0 / 0 / 0 |

## 所見

### 良い点

- exp345のpooled改善という直接根拠があり、変更をforward後のbackward smoothingだけに限定できる。
- exp345保存parent/causal predictionを使うため、追加control HMMコストがない。
- future raw GR利用はコンペ推論契約と一致する。

### リスク

- 誤ったbase pathへのcalibrationを未来GRがwell全体へ逆伝播し、tailをさらに悪化させる可能性がある。
- offline限定で、online/streaming用途には一般化できない。
- exp345のhidden-like欠落を繰り返さないため、2 scopeの存在自体を必須gateにする。

## 実行結果

- parent `14.501048` → candidate `14.367548`: `+0.133499 ft`、5/5 folds改善。
- causal `14.331543` → candidate `14.367548`: `-0.036006 ft`、2/5 folds改善。
- hidden-like 2面は両baseline比で非悪化。
- parent比by-well median `-0.008672 ft`に対しp95 `+1.346427 ft`、worst `8995c945`は`+20.887374 ft`。
- technical gateは全PASSしたため、実装失敗ではなく科学仮説のFAILと判断した。
- decision: `stage_0_failed_close_without_rescue`。

## 実行入口

- train notebook: `exp350_exp345_bidirectional_gr_affine_smoother_train.ipynb`（compact self-contained Stage 0実装）
- inference notebook: `exp350_exp345_bidirectional_gr_affine_smoother_inference.ipynb`（scaffoldのみ）
- canonical kernel: `kentookumura/exp350-bidirectional-gr-affine-smoother-train` version 1。
- Stage 1、inference、submission、post-hoc救済は行わない。

## 次

branchを閉じる。full-well smoothingは平均ではparentを改善したがcausalを下回り、tailを抑える目的に反してworstを拡大した。同familyの救済候補は追加しない。
