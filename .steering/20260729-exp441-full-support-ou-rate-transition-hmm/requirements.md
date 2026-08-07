# 要件

## 依頼

rateを使いながらexp209の追従遅れを減らす第1案として、全support
Gaussian/OU rate transitionを`exp441_full_support_ou_rate_transition_hmm`
として設計確定する。バックログ、steering、実験scaffoldと記録文書を作る。
当初は科学ロジック、test、実行可能Notebook、Kaggle package/runを未実装とした。
2026-07-29のユーザー依頼「exp441を実装してください」により、compact
self-contained候補とcontract testの実装を承認済みとした。2026-07-30の
ユーザー依頼「実行してください」により正規train Notebook採用、Kaggle
package、Stage 0 runも承認し、version 1を完走した。Stage 1、inference、
submissionは未承認のままとする。

## 仮説

exp209はrateを先に更新し、更新後rateでTVTを進めているが、rate kernelが
隣接3状態に制限される。親の`momentum`と`sig_r`から定まるexact OU conditionalを
全rate binへ積分すれば、新しい調整値なしに有限伝播速度を除去できる。

Assumption: exp209の遅れの一部が、GR識別力不足だけでなくtri-diagonal離散化に由来する。

## 制約

- Route: `pf_beam`。
- 親/control: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- 変更はrate transition kernelだけ。
- `momentum=0.998`、`sig_r=0.002`、per-well 41-state rate gridを再利用する。
- TVT transition、position noise/grid/band、GR emission、prior、missing処理、
  forward/backward、posterior-mean readout、境界mass切捨ては固定する。
- trigger、jump mixture、reset、re-anchor、acceleration、blend、selectorを使わない。
- same-OOF parameter/gate rescueをしない。
- compact候補、contract test、正規train Notebook採用、Kaggle Stage 0を実施済み。
- Stage 1、rerun、inference、submissionはStage 0 FAILにより閉鎖。

## 受け入れ基準

- OU平均・分散、bin積分、finite-support外massの扱いが一意である。
- fixed32 1候補×32 wells、保存parent rerun 0が固定されている。
- technical、mechanism、Stage 1のAND gateとfail actionが固定されている。
- truth-lateとSHA契約が記載されている。
- model / booster / PF / Beam / GPUが0と記録されている。
- compact self-contained train/inference候補と専用testが静的検証を通る。
- Stage 0結果を固定gateで判定し、FAILならStage 1以降がfail-closedになる。
