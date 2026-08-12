# exp428_similar_well_gr_registration_map_transfer_readout

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 technical FAIL / no-rescueで閉鎖
- Stage: 0-model train-side OOF readout
- CV / LB: technical coverage不足のため無効・対象外
- 作成日: 2026-07-28
- 親実験: `exp423_same_typewell_gr_dtw_truth_warp_transfer_readout`
- steering:
  `docs/legacy/steering/20260728-exp428-similar-well-gr-registration-map-transfer-readout/`

## 仮説

同じType Well GR波形を共有するwellの中では、Horizontal GR波形が似たdonorで測った
`Type Well GRを真のTVTから何ftずらすとHorizontal GRに一致するか`という
registration offsetがheld-out queryへ転送できる。

## exp423との違い

exp423が転写したのはdonorの正解TVT増分/pathだった。本実験はdonor自身の
Type Well–Horizontal GR対応から得るoffsetだけを転写する。donorの絶対TVT、TVT増分、
geometry、rateはquery出力へ使わない。

## 固定した設計

- `exp065 native_overlap=1`をsame-Type-Well eligibilityにする。
- donorはouter-trainだけ、queryはouter-validだけとする。
- Horizontal suffix GRの固定constrained DTWで似たdonorを順位付けする。
- donor true TVTの周りで固定13 shiftを比較し、512-row blockごとのbest ZNCC shiftを得る。
- primaryはrank-1 donorのglobal median shift（ft）。
- local offset曲線、stretch、local warpはmapping shapeの追加診断だけに使う。
- row lagはft補正に使わず、exact-overlap部分のType Well TVT差で軸を変換する。
- query true TVTはcandidate/artifact freeze後の参照shift作成と評価にだけ使う。
- TVT candidate、LightGBM、HMM、PF、Beam、inference、submissionは作らない。

## 検証方針

- Fold: `exp423` / `exp109`と同じwell単位5-fold。
- Primary: rank-1 donor global shiftのequal-well MAE。
- Controls: zero shift、stable-random donor、same-group median、post-freeze top-5 oracle。
- Leakage: donor/query交差0、query truthのfreeze前read 0、Type Well軸graph conflict 0。
- Gate: coverage、oracle headroom、primary対3 controls、GR距離–誤差相関、ZNCC gain、
  hidden-like、by-well tailを事前固定ANDで判定する。

## 実行量

- audit variant: 1
- reporting folds: 5
- LightGBM config / trained fold / booster: `0 / 0 / 0`
- PF / HMM / Beam well-run: `0 / 0 / 0`
- GPU / parent-control replay: `0 / 0`

## 実装状態

compact self-contained Jupytext train source、正規train notebook、Type Well軸graph、
registration map、constrained DTW、target-free freeze、late query truth join、global/local
gate、生成物保存を実装した。専用test 15件、構文、F821、Jupytext round-trip、
strict experiment validationは通過済み。

正規inference notebookは対象外のplaceholderとして維持した。Kaggle CPU version 2まで
実行済み。既存emission/candidateへの統合、inference、submissionは行わない。

## 所見

version 1はDTW欠損処理バグでsupport 0となり、親互換の決定的補間へ最小修正した。
version 2はsupported `306 / 773 = 39.586%`で固定下限70%をFAILした。評価可能な290 wells
でもprimary shift MAE `2.529310 ft`はzero `1.105172 ft`より`1.424138 ft`悪く、
top-5 oracleもzeroより`0.013793 ft`悪かった。technical/scientific/local gateは全FAILし、
`invalid_or_insufficient_registration_support`でbranchを閉鎖した。

設計時のtarget-free確認では、exp065のexact native-overlap edge 10,697件のType Well
TVT軸差は全て0 ftだった一方、10,656件はrow lagが非ゼロだった。このためrow lagを
registration shiftと解釈せず、TVT軸差を明示的に検証する設計としている。

同じOOFでsupport、group、DTW、shift grid、blockを救済せず、independent rerun、
inference、submission、HMM/PF/Beam統合へは進めない。
