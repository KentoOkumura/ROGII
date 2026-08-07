# exp321 条件付き後続案の固定契約

この文書は、別セッションで案4/5の内容がexp321へ混入したり、別の自由度へ変形されたりすることを防ぐ。両案とも未採番・未実装であり、exp321のStage A/B/Cには含めない。Run AB version 1でStage Bが固定bank range/quantization gateをFAILしたため、開始条件は成立せず、両案を救済なしで閉じた。

## 案4: z_only_residual_offset_exact_hmm_probe

- 状態: 閉鎖。exp321 Stage B FAILにより開始条件不成立。
- 開始条件: exp321 Stage BとStage C科学的gateの両方がPASS。
- base: `tvt_z`。状態は`tvt=tvt_z+delta`。
- decoder: exp281のoffset exact HMMを1 variantだけ固定継承する。
- 固定値: `delta [-80,80] ft`、step `0.35 ft`、41 rate states、rate span `±0.10`、`sig_r=0.002`、`sig_p=0.02`、momentum `0.998`、exp209 Gaussian raw-GR emission。
- 比較: exp321 `tvt_z` / `tvt_z_gr`、exp281 residual-offset HMM、exp226 final、exp263 fixedの保存済み予測・score。
- 禁止: grid、rate、process noise、likelihood weight、sigma、blendの同一truth救済、PF/Beam、inference、submission。
- 主リスク: exp281で発生したp95/worst-wellの大幅悪化をZ-only中心でも再発すること。

Run ABのshift bank拡張、sigma変更、threshold緩和を用いてtriggerを事後救済しない。新しい独立根拠がない限り再開しない。

## 案5: z_only_gr_sparse_candidate_addonly

- 状態: 閉鎖。exp321 Stage C未到達により開始条件不成立。
- 開始条件: exp321 Stage Cが全科学的gateをPASSし、`tvt_z_gr` RMSEが保存exp226 final `9.427110`以下。
- selectorへ追加するcandidate: `tvt_z_gr` 1本だけ。`tvt_z`は診断controlでありcandidate bankへ追加しない。
- Stage 0: exp293/修正版exp264 fixed12に対するH512/whole-well add-one oracle、strict unique-best、残差相関を0 modelで監査する。
- Stage 0 gate: H512とwhole-well oracle gainが各`>=0.05 ft`、5/5 folds、strict unique-best block/well比の少なくとも一方が`>=5%`。
- Stage 1: Stage 0 PASS時だけ、exp286と同じ13番目add-only contractで候補ID/order、generic proxy、native confidence、exp264 outer5/inner4 dual objectivesを維持し、保存済みexp264 controlを再学習せず新selectorだけを学習する。
- downstream: selector guard PASS後にのみ検討し、model数、fold数、booster数、GPU/CPUコストを明示して別承認を得る。
- 禁止: `tvt_z`と`tvt_z_gr`の2候補同時追加、hard replacement、truth-derived router、threshold/weight grid、control再学習、先行gate前のinference/submission。

triggerを満たさなかったため、candidate追加へ進むためのcap/window/shift救済は行わない。新しい独立根拠がない限り再開しない。
