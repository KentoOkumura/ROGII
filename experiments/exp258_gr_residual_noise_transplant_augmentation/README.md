# exp258 GR residual noise transplant augmentation

実測 horizontal GR から typewell の affine 再構成で説明できない残差を取り出し、fold-safe な
連続 block として nested candidate ranker の学習行へ移植する実験です。ranker の再学習後、
outer-fold safe な rank-slot 特徴を exp218 の特徴面へ add-only し、最終 TVT LightGBM も
再学習します。

## 状態

Kaggle selector train v1が完了しました。primary `real_residual_block` ranker 20 CPU boostersは
完走しましたが、historical exp238に対する6 guard中5項目が不通過でした。expected-error MAEだけは
`4.532978 -> 4.523354`へ改善しましたが、global / near / 1000+ / candidate AUC / worst-wellが
悪化したため、final TVT LightGBM 15 GPU boosters、inference、submissionは実行しません。

## 親実験と変更点

親実験は`exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`です。11 candidate、
HMM/PF/Beam、nested split、後段LightGBMの構成は固定し、selectorのinner-train行だけを
実測GR残差blockで拡張しました。historical exp238を比較対象とし、親/controlは再学習していません。

## 仮説

実測 GR の欠損、spike、周期、高周波、量子化を保った残差 block を inner-train 行へ追加すれば、
iid Gaussian noise より実環境に近い candidate-likelihood の揺らぎを ranker が学習できます。
その clean nested OOF score を最終 TVT LightGBMへ渡すことで、exp238より安全な rank-slot
confidenceを得られると考えます。

## データ拡張の定義

この実装で増やすのは selector の inner-train candidate-long 行です。recipient の true TVT と
typewell GR から作る clean GR に、別の inner-train well の実測 residual blockとmissing maskを
移植します。その synthetic GR から target-freeな `multiobs_*` likelihood列だけを再計算し、
元のclean行と合わせてrankerを学習します。

candidate TVT path、HMM emission/transition、PF/Beam生成、正解ラベルは改変しません。
inner-valid / outer-valid は元のclean featureのままです。`clean_duplicate`は元の実測GR行の複製、
`white_noise`と`shuffled_residual`は別runのnegative controlで、primaryと混ぜません。

## 検証方針

- exp238と同じ11 candidate、outer 5 × inner 4 well GroupKFold、expected-error rankerを固定します。
- donorは各inner modelのinner-train wellだけに限定し、inner-valid / outer-validとのwell重複をassertします。
- historical exp238 nested scoreとcandidate AUC、expected-error MAE、global、near、1000+、worst-wellを同一foldで比較します。
- primary selectorが全guardを通った場合だけ、exp218 380特徴 + rank-slot 35特徴で3 config × 5 foldを学習します。
- final OOFはexp238とoverall、6 distance bucket、2 hidden-like subset、5 fold、by-wellで比較します。
- inferenceではaugmentationを行わず、保存済み20 selectorと15 final modelだけをclean current testへ適用します。

## 所見

実測残差block augmentationはexpected-error calibrationをわずかに改善しましたが、candidate選別と
direct safetyは一貫して改善しませんでした。historical exp238比でglobal `+0.004554`、near
`+0.001652`、1000+ `+0.006871`、worst-well `+0.322063` ft、candidate AUC
`-0.000175`です。GR残差をHMM emissionへ直接注入せずrankerだけを頑健化する設計でも、後段TVT
LightGBMを再学習するだけの安全性根拠は得られませんでした。

## 実行コスト契約

- Stage 0: 0 variant / 0 config / 0 fold / 0 booster。
- Stage 1 primary: 1 variant × 1 config × outer 5 × inner 4 = 20 CPU boosters。
- Stage 1 negative controlsを全て実行する場合: 4 variants合計80 CPU boosters。既定では実行しません。
- Stage 2 primary: 1 variant × 3 configs × 5 folds = 15 GPU boosters。
- historical exp238 / parent / controlの再学習: なし。

Kaggle push前に対象stage、variant、booster数の明示承認が必要です。

## 次アクション

この分岐はnegative resultとして終了します。guard契約に従いfinal学習や救済gridへは進まず、
exp258由来の追加実行、inference、submissionはありません。
