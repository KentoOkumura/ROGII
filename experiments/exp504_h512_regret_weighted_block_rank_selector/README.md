# exp504_h512_regret_weighted_block_rank_selector

## 状態

- Route: `ensemble`
- 状態: train完了・technical PASS / scientific FAIL・terminal close
- 親: `exp293_physics_only_candidate_bank_headroom_contract`
- feature契約: `exp264_exp263_candidate_confidence_dual_selector` corrected 88列
- CV: `8.114276980`、anchor比`-0.124054566 ft` / Public LB・Private LB: 未提出
- 作成日: 2026-08-02

## 仮説

行ごとのabsolute-error予測ではなく、H512 blockをqueryとして12候補の相対順位を直接
学習すれば、長区間で安定して誤った候補を選ぶ失敗を減らせる。

## 確定した変更

- exp293 fixed12 candidate bankと保存H512 blockをそのまま使う。
- corrected exp264 88列をblock内で固定9統計へ集約する。
- block内candidate pairの勝敗を、block MSE差に基づくregret weightで学習する。
- 両方向確率を反対称化し、Borda scoreで1候補を選ぶ。
- 非anchor候補は固定anchorへの勝率が0.5を超える場合だけ採用する。
- 選択候補をblock全行へ適用し、元のrow RMSEで評価する。

## 固定した範囲

- H512のみ。H128/H256/whole-well/overlap/可変長は扱わない。
- 1 variant × 1 rank config × outer 5 folds = 5 CPU models。
- 親control再学習、候補再生成、PF/HMM/Beam、GPU、inference、submissionは0。
- loss、weight、model、guard thresholdの探索を行わない。

## 検証方針

exp293のwell-grouped outer 5 foldsを再利用する。outer-train truthだけでpair labelとweightを
作り、outer-validではrank predictionをSHA固定してからtruthを評価用に読む。primaryは
元のrow-level OOF RMSEで、fixed anchor `8.238331546`をmatched controlとする。

## 科学的PASS

すべて満たす場合だけPASSとする。

- fixed anchor RMSE `8.238331546`から0.05 ft以上改善。
- 4/5 folds以上で非劣化。
- 固定5 scopeがすべてanchor比 `+0.02 ft`以内。
- by-well delta p95 / worstがともに `+0.25 ft`以内。
- technical gateをすべて通過。

## 実行入口

compact self-contained train候補を正規train notebookへ採用した。正規inference notebookは
markdown-only placeholderのまま維持する。Kaggle private CPU version 1
（`id_no=129488458`）が完走した。

実装は次を含む。

- exp263保存cacheからexp293 fixed12を数値互換で再構成し、candidate/block/schema SHAを照合。
- corrected exp264 88列をtarget-freeに再生成し、`ctx__` 22列をshared、残り66列を
  candidate-specificとしてH512内9統計へ集約。
- 1,986列ordered-pair表現、outer-train truthだけのlabel/regret weight、5 CPU boosters。
- 両方向確率の反対称化、Borda、0.5固定anchor guard、row-level OOF評価。
- fold/scope/by-well/rank/switch/choice/feature importanceと再現性SHAの保存。
- 9件のcontract test、Jupytext round-trip、構文/F821、strict experiment validation。

## 所見

technical gateは全PASSし、pooled OOFはanchorから`0.124055 ft`改善した。しかし非劣化は
`3/5 folds`、hidden-like spatial / typewell-purgedは`+0.285759 / +0.269833 ft`、by-well
p95 / worstは`+2.963656 / +16.799044 ft`で固定gateをFAILした。exp504はsame-OOF救済、
inference、submissionなしで終端閉鎖する。

詳細な数式とtruth-late順序は
`docs/legacy/steering/20260802-exp504-h512-regret-weighted-block-rank-selector/design.md`を正とする。
