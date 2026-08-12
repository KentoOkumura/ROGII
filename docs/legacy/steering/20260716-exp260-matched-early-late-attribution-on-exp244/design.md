# 設計

## アプローチ

exp244 v4で検証済みのofficial cacheと4 offset cacheを再利用し、official rowsは両variantで共通にする。
early-onlyは`m1000/m250`の384,250 pseudo rows、late-onlyは`p250/p1000`の385,907 pseudo rowsだけを
weight 0.5で追加する。各variantをexp218と同じ3 config x 5 foldで独立学習し、保存済みexp218 OOF、
保存済みexp244 mixed OOFと同一official surfaceで比較する。

feature cacheは1回だけmemmapへstreamし、variantごとにoffset maskを適用する。各foldではouter-valid
source wellをpseudo trainから除外する。両variantのvalidationはclean official-startのみである。

## 実験範囲

- 対象実験: `exp260_matched_early_late_attribution_on_exp244`
- Route: `ml_model`
- 親実験: `exp244_bidirectional_prediction_start_pseudotail_augmentation`
- 変更する変数: pseudo direction (`early_only`, `late_only`)
- 固定する変数: official/pseudo cache v1 SHA、row sampling、row weights、380 features、exp218 3 config、
  5 folds、early stopping、GPU deterministic mode、official-start validation、stress surfaces、adoption guards

## 再現性設計

- seed policy: exp218 `gpu_repro_guard_dp_threads8`の固定seed / modeをそのまま使う。新規乱数処理なし。
- stochastic 処理の有無: 新規なし。特徴量cacheはexp244 v1の決定的生成物をSHA pinして読む。
- PF/Beam / likelihood-PF / seed bagging の有無: 新規生成なし。保存cache内の既存特徴量だけを使う。
- 並列処理と乱数の関係: LightGBMの固定seedと`num_threads=8`をexp218 configから継承する。
- CPU/GPU runtime と deterministic flags: Kaggle GPU、internet off、`gpu_use_dp=true`、
  `deterministic=true`、`force_col_wise=true`を親modeからhard assertionする。
- train cache / test feature regeneration の SHA 記録方針: officialと4 pseudo cacheのmanifest/schema/
  request/content SHAをexp244値へpinする。test featureは生成しない。
- model manifest / prediction / submission SHA 記録方針: variantごと15 modelと全体manifest SHA、OOFは
  decompressed SHAを保存する。submissionは生成しない。
- Kaggle package bootstrap 確認方針: prepared notebookのbootstrapからconfigと親helperを復元し、
  experiment name、run approval、2 variants、30 boosters、7 kernel sources、GPU/internet metadataを確認する。
- exp244 v4自体はrerunしていないため、本実験をdeterministic anchorとは呼ばない。

## リスク

- リークリスク: late rowsはtrain-only。outer-valid source well由来pseudo rowsを除外し、validationへpseudo rowsを入れない。
- CV/LB 不一致リスク: raw testではlate viewを作れず、train-side attributionだけで推論可能性は主張できない。
- ランタイム/メモリリスク: 30 boostersはexp244の2倍。memmapを共用し、variantごとのtrain/valid matrixをfold終了時に削除する。
- 再現性リスク: GPU LightGBMはbitwise一致を保証しない。input/model/prediction SHAを記録し、採用候補化する場合のみrerunを相談する。
- 解釈リスク: exp239 early-onlyとはsamplingが異なるため、本実験内のmatched結果だけで方向帰属を判断する。
