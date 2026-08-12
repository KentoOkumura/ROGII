# 要件

## 依頼

閉鎖済み`exp316_typewell_group_candidate_family_error_prior`の中核仮説を、
`exp311` / `exp312` / `exp313` / `exp315`に依存しない独立Stage 0として新番号へ切り出す。
固定exp293 candidate familyについて、Type Well群ごとのfamily error priorがheld-out wellの
family順位を再現するかを、selector学習前の0-model readoutで判定する。

当初は設計確定までとしていたが、2026-07-23のユーザー依頼
「exp354を実装してください」により、Stage 0 prior generator、stable shuffle、
compact self-contained Notebook候補、contract testsだけを実装対象へ変更した。
2026-07-23の追加依頼「実行してください」により、compact train候補の正規Notebook採用、
Kaggle CPU package/push/run、Stage 0完了監視までを実行対象へ変更した。
Stage 1 selector 40 models、inference、submissionは引き続き未承認とする。

Stage 0 version 1はreal-minus-shuffle Spearman `-0.001290`で固定gateをFAILしたため、
Stage 1は未承認に加えて不適格となった。同じreadoutでの救済gridや再実行は行わない。

## 制約

- Route: `ml_model`。
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 科学的入力は`exp293_physics_only_candidate_bank_headroom_contract`の固定deployable12
  candidate/family manifestとOOFだけとする。
- downstream controlは保存済みcorrected `exp264`とし、Stage 0ではselectorを学習しない。
- exp311/312のgroup calibration/emission、exp313 guard output、exp315 rank featureは入力禁止とする。
- outer-train wellsだけでgroup×familyのwell等重みMAE/RMSE/best-family率を推定する。
- supportは`k=10 wells`でglobal family priorへ固定縮約し、未seen groupはglobal、supportなしはneutralへ落とす。
- well ID prior、hard family router、candidate固有threshold、候補値変更、oracle ruleは禁止する。
- Stage 0は1 prior / 1 group-label shuffle / 5 folds / model・booster各0。
- Stage 1はStage 0全PASSと別承認時のみ、corrected exp264へadd-onlyする
  1 variant × 2 objectives × 5 outer × 4 inner = 40 selector models、control再学習0とする。

## 受け入れ基準

- candidate/family/fold identity parity、全prior finite、held-out group coverage `>=0.90`。
- Stage 0のheld-out family rank Spearman `>=0.15`、正方向 `>=4/5 folds`。
- real minus group-label-shuffle Spearman `>=0.05`。
- hidden-like spatial / typewell-purgedの両面でSpearman非負。
- Stage 0 PASS時だけStage 1実装を別承認し、saved corrected exp264比 `>=0.03 ft`改善、
  `>=4/5 folds`、hidden-like非悪化、worst `<=+0.25 ft`を要求する。
- deterministic readoutとして扱う場合は、candidate/family/fold/prior/fallback/readoutのcontent SHAを記録する。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 2026-07-23 実装境界

- Stage 0 primary rankは、shrunk family RMSEとheld-out family RMSEのwell内Spearmanを
  well等重みで集約する。
- family内に複数candidateがある場合はcandidate等重み、group/global集約はwell等重みとする。
- exp293 primitive 6本はexp263 catalogのsource family、固定formula 6本は
  exp263 loaderの`virtual_combination` familyを継承する。
- MAEとbest-family率も固定priorとして保存するが、同じreadoutを見た後のmetric選択は行わない。
