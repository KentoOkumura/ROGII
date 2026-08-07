# 設計

## アプローチ

各outer foldで利用可能なpeer数・support・group identityをtruth-freeにfreezeし、(1) exact native-overlap group、(2) exp319が将来PASSした場合のみsoft similarity、(3) global prior、(4) identityの固定fallbackを監査する。現時点のdefaultはidentityで、soft pathはdisabled。same-group gainとunseen/purged non-regressionを別々に評価する。

## 実験範囲

- 対象: `exp313_typewell_group_unseen_transfer_guard`
- Route: `pf_beam`
- 親: `exp312_typewell_group_conditional_gr_emission_table`
- 変更: calibration値ではなく利用可否とfallbackだけ。
- 固定: group identity、peer/support threshold、fallback順、3 split surfaces。
- 計算量: 3 audit surfaces、5 folds、model/booster/decoder 0。

## 再現性設計

- stable group/peer ordering、SHA256 fold、deterministic fallbackを使う。
- group membership、availability、fallback reason、prior source、score tableのcontent SHAを保存する。
- identity fallback parityをrow単位でassertする。

## リスクと停止条件

- 同群で良くてもhidden testに同群peerが存在しない可能性がある。
- coverage 90%、same-group gain 0.05 ft、4/5 folds、unseen non-regression、worst +0.25 ftの全条件を要求する。
- FAIL時はthresholdやfallbackを同一OOFで調整せず、group prior downstreamを停止する。
