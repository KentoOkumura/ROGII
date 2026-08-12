# 設計

## アプローチ

exp482のStage A0で得た「763/773 wellsにraw shockがありzero-shock wellは10」
という事実を受け、対照wellを使わず、同じtarget-free順序で選んだshock-support
top32だけを評価する。各well内では保存済みexp209予測をparentとして、同じ行の
candidateと直接比較する。

## 実験範囲

- 対象実験: `exp488_isolated_gr_shock_prior_hold_support_only`
- Route: `pf_beam`
- 親実験: `exp482_isolated_gr_shock_prior_hold`
- 科学的親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: Stage 0 manifestをsupport32 + control32からsupport32だけへ変更。
- 固定する変数: raw shock全条件、HMM state/grid/transition/emission/prior、
  LOO式、trigger閾値、saved parent、support ordering、科学gate。
- 実行量: 1 candidate、773-well raw census、32 parent-message HMM replay。
  candidate state HMM / parent prediction rerun / model / booster / PF / Beam / GPUは0。

## 再現性設計

- seed policy: RNGなし。well、row、state、message、manifest順を安定化。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: single worker / single numba thread、乱数なし。
- CPU/GPU runtime と deterministic flags: Kaggle private CPU、GPU/internet無効。
- train cache / feature SHA: raw census、raw-shock rows、support32 manifest、
  message、trigger、predictionのlogical/decompressed SHAを記録。
- model manifest / submission SHA: model/submissionを作らないため非該当。
- Kaggle bootstrap: 埋め込みconfigのauthorization、32 replays、GPU/internet無効、
  exp209/exp226 kernel sourceをpush前に確認する。

## リスク

- リークリスク: shock countによるtop32選択はtruth/fold/errorを読まずfreezeする。
- CV/LB 不一致リスク: shock-enriched supportだけなので全OOFやLBへ一般化しない。
- ランタイム/メモリリスク: exact HMM 32 wells。投影30,600 sec、RSS 25 GBを上限。
- 再現性リスク: 初回成功runをdeterministic anchorにはしない。
- 解釈リスク: zero-shock対照群を外すため、非発火wellでの群間安全性は主張しない。
  row-local不変性は実装test、support内tailはby-well gateで確認する。
