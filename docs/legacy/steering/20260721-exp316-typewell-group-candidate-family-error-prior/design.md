# 設計

## アプローチ

各outer foldのtrain wellsで、固定exp293 candidate familyごとのsuffix MAE/RMSE/best率をwell等重み集計する。native-overlap群統計を`k=10 wells`でglobal familyへ縮約し、候補rowへfamily priorとして付与する。Stage Aではheld-out family performanceとの順位相関だけを読む。PASS時のみexp315またはsaved corrected exp264のnested selectorへadd-onlyする。

## 実験範囲

- 対象: `exp316_typewell_group_candidate_family_error_prior`
- Route: `ml_model`
- 親: `exp315_typewell_group_candidate_likelihood_rank_features`
- 変更: group×family soft priorだけ。
- 固定: candidate bank/family、outer5/inner4、2 objectives、fallback、gate。
- 計算量: Stage A 0 model、Stage B 40 selector models。

## 再現性設計

- candidate family manifestとouter/inner fold SHAをhard preflightする。
- prior input/output schema/content SHA、fallback reason、Stage A readout SHAを保存する。
- Stage Bではmodel manifest、各model SHA、OOF prediction SHAを保存する。

## リスクと停止条件

- 群別誤差priorはlabel-derivedなのでcross-fitを一段でも省略すればleakとなる。
- group内well数が小さい場合は過信せずglobal/neutralへ戻す。
- Stage A相関またはfold安定性FAILならselector学習せず閉じる。
