# 要件

## 依頼

実装済みだが未実行で閉じた`exp308_imputed_gr_confidence_downweight`をreopenせず、
失敗したexp307 finite-MAD observationへの依存を削除した新番号の独立実験を設計する。
今回は設計とscaffoldだけを作成し、旧コード移植を含む実装は行わない。

## 2026-07-25 追加依頼

ユーザーの「exp358を実装してください」をStage 0実装承認として扱う。
承認範囲は0-HMM technical auditのcompact self-contained train候補と
fail-closed inference候補までとし、正規Notebook採用、Kaggle package/push/run、
Stage 1、inference、submissionは含めない。

## 2026-07-25 Stage 1追加承認

ユーザー指示「Stage 1に進んでください」を、Stage 0 technical PASS後に予約していた
train-side Stage 1の実装、正規train Notebook採用、Kaggle private CPU package/push/run、
完了監視の承認として扱う。

承認量はfixed `missing_distance_half_life_8_floor_0p25` 1 variant、
5 reporting folds、773 exact-HMM well-runs、model / LightGBM config /
trained fold / booster / PF / Beam / parent-control再実行各0。
inferenceとsubmissionは承認に含めない。

## 制約

- Route: `pf_beam`
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 履歴参照: `exp308_imputed_gr_confidence_downweight`
- exp209のGR interpolation、zero-filled residual sigma、typewell、grid、transition、
  prior、posterior meanを固定する。
- 唯一の変更はraw finite GRでないrowのGaussian log emissionへ
  `max(0.25, 2^(-distance/8))`を1回だけ掛けること。
- observed rowはweight exact 1。zero weight、hard mask、half-life/floor gridは禁止。
- Stage 0はweight schedule technical audit 1、HMM/model/trained fold/booster各0。
- Stage 1はStage 0 technical gate PASSと別承認時だけ1 variant / 773 HMM runs。
- exp209 controlは保存成果物を使い再実行しない。

## 受け入れ基準

- raw missing mask、nearest-finite distance、weight、interpolated GRをtruth結合前にfreezeする。
- Stage 0で773 wells / 3,783,989 rows、全finite、observed exact 1、
  missing weight範囲`[0.25,1)`、nontrivial missing weightを確認する。
- Stage 1でexp209比0.05 ft以上、4/5 folds、1000+・hidden-like・p95・worst guard、
  fixed LikPF 50:50非悪化を要求する。
- parent/control再実行0、GPU/internet off、inference/submission別判断。
- Stage 0 technical PASSを入力条件とし、Stage 1 prediction/weight surfaceを
  truth結合前にSHA freezeする。
- Stage 1の科学gateは事前固定値から変更せず、FAIL時は同一OOF rescueを行わない。

## 2026-07-25 完了判定

Kaggle private CPU version 2でStage 1を完了した。candidate RMSE
`12.012569787442315`はexp209 control `11.938287234887435`より
`0.07428255255488025 ft`悪化し、改善fold 0/5、required scope、
by-well、fixed blend gateもFAILした。事前契約どおりrescue、再実行、
inference、submissionを行わずclosedとする。

formal technical gateの`missing_weight_formula_exact=false`は、frozen gzip CSV
再読込後の753 rowsにおける最大`5.551e-17`のround-trip差だった。
この事後監査結果で事前gateは書き換えず、科学FAILの判定も変更しない。
