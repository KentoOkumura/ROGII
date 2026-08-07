# exp304_gr_denoiser_emission_separability_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU v1完了、quality gate PASS、`swt_db4_l3`選択
- CV / Public LB / Private LB: なし / なし / なし
- 作成日: 2026-07-20
- 親実験: なし。方法論はexp280、emissionはexp209、negative evidenceはexp189を参照する。

## 仮説

HMM/PFへ平滑化GRを直接入れる前に、固定shift候補のtruth-nearest順位を調べれば、ノイズ低減と
wrong-mode安定化を分離できる。rawと同じ候補、block、sigma、missing policyを使ったとき、
robust RTS、stationary wavelet、L1 trendのいずれかがMRR/top3を安定して改善するかを判定する。

## 固定した設計

- exp280と同じexp226 `tvt_geop`、13 shift、非重複512行block、5 foldを使う。
- `raw / robust_rts / swt_db4_l3 / l1_trend`の4 signalを比較する。
- known-prefix raw sigma、clip、GR missing補間を共有し、denoised GRからsigmaを再推定しない。
- scoreとcontent SHAをtruth-freeで凍結した後だけtrue TVTを付与する。
- primaryはblock MRR/top3。pooled各`+0.01`、4/5 folds、1000+、hidden-like 2面、sharp-edge、
  shuffled control、decoy gapの固定gateを全て通った方式だけを後続候補にする。
- Late専用scopeは作らない。rolling median / Savitzky-Golayはexp189のnegative resultを採用し再試行しない。

詳細は[steering design](../../.steering/20260720-exp304-gr-denoiser-emission-separability-readout/design.md)を正とする。
案2〜4の開始条件と禁止事項は[reserved follow-up contract](reserved_followup_contract.md)を正とする。

## 実装

- 別名Jupytext source / Notebook:
  `exp304_gr_denoiser_emission_separability_readout_compact_selfcontained_train.py/.ipynb`
- fail-closed inference source / Notebook:
  `exp304_gr_denoiser_emission_separability_readout_compact_selfcontained_inference.py/.ipynb`
- rawと3 denoiser、solver status、denoised series streaming freeze、target-free score freeze、late truth join、
  scope/fold metrics、fixed technical/quality gate、expected生成物保存を実装した。
- ユーザーの実行承認によりcompact trainを正規train Notebookへ採用する。inferenceは引き続きfail-closed guardとする。
- 2026-07-20に正規train採用とprivate CPU packageのstrict preflightを完了した。

## 実行量

- Kaggle CPU v1: 4 variants、13 shifts、保存済み5 fold strata、model 0、LightGBM config 0、trained fold 0、booster 0
- HMM / PF / Beam well-run: `0 / 0 / 0`
- GPU / inference / submission: なし / なし / なし

正規trainは承認済みcompact self-contained実装へ採用する。正規inferenceは成果物を書かず必ず停止するguardのまま維持する。

## 検証方針

- Fold / Group: exp280と同じ保存済み5 folds / `well_id`。
- 主評価: 非重複512行blockのMRRとtop3。raw比とstable shuffled比を同時に読む。
- Leakage check: denoised seriesと全target-free scoreのcontent SHAを凍結するまでtrue TVT、error、
  formationをscore側へ渡さない。
- Promotion: steering/configに固定したpooled、fold、1000+、hidden-like、sharp-edge、decoy-gapの
  全gateを満たす1方式だけを選ぶ。結果後のgrid救済はしない。

## 結果

Kaggle private CPU version 1（id_no `128011752`）を3,783,989 rows / 773 wells / 7,787 blocksで完了した。
rawのMRR/top3 `0.389626 / 0.452421`に対し、`swt_db4_l3`は`0.424724 / 0.504687`で、
`+0.035098 / +0.052267`改善した。MRR/top3とも5/5 foldsで改善し、必須4 scope、shuffled control、
decoy-gapを含む全quality gateをPASSした。

technical gateはrawとSWTが全1,546 seriesでPASSした。robust RTSは1,531 failures、L1 trendは974 failuresで
technical FAILとし、部分scoreを科学的比較には使わない。silent fallbackは0である。

## 所見

stationary db4 level-3 SWTは、raw sigmaを共有した固定shift emissionの識別性を一貫して改善した。
一方、これはdecoder RMSEやLBの改善証拠ではない。RTS/L1の反復・閾値救済は行わず、SWTだけを後続へ渡す。

## 次

exp304は完了とする。予約契約の案2`tempered_raw_smoothed_exact_hmm_emission`は開始条件を満たしたため、
別exp候補としてbacklogへ移す。SWT選択のためRTS variance専用の案3は閉じ、案4は案2 PASSまで開始しない。
