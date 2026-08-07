# 設計

## アプローチ

HuberはGaussianと同じ中心二次損失を保ち、外れ値だけを線形penaltyへ変える。Student-tがtailを救うが全体を平坦化しすぎるという事前条件が成立した場合だけ、より保守的なrobust familyとしてexp280 rank auditを行う。

## 実験範囲

- 対象実験: `exp344_exp226_huber_residual_offset_emission_audit`
- Route: `pf_beam`
- 親: Stage 0 exp280、Stage 1 exp281。
- dependency: exp342 Stage 0のpartial pattern。
- 変更する変数: Gaussian loss→Huber `delta=1.345`だけ。
- 固定する変数: sigma、missing、shift/block、exp226 path、HMM state/transition/prior/output。
- 実行量: Stage 0 HMM 0、Stage 1条件付き1 variant / 773 runs、control再実行0。

## 検証方法

1. exp342 dependency summaryとSHAをpreflightする。
2. exp280 Gaussian scoreを再現し、Huber block scoreをtruthなしでfreezeする。
3. rank/stress/shuffle/extreme-residual gateを判定する。
4. PASSと別承認時のみexp281 grammarでfull HMMを実装する。

## 再現性設計

- RNGなし。shift/block/well順固定。shuffleはstable SHA256 rotation。
- CPU、GPU/internet off、Stage 1最大8.5時間。
- dependency/input/score/prediction/metricsのdecompressed content SHAを記録する。
- model/submission SHA非該当、inference/submission disabled。

## リスク

- repeated testing: exp342の事前登録partial patternだけをtriggerにし、自由なrobust family探索をしない。
- alias flattening: Student-tより弱いがwrong state penaltyも抑えるためStage 0 rank gateを必須とする。
- implementation risk: Huber code pathは新規だがdelta 1設定以外を許可しない。

## 優先度

条件付き`P4`。Student-tの結果なしに実装しない。
