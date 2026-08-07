# exp310_effective_sample_size_shrunk_sigma_gr セッションノート

## 目的

exp307のwell別finite-MAD `σ_GR`を、有効標本数に応じてwell間priorへ縮約する案を、実装前に単一式と実行triggerまで固定する。

## 現在の状態

- Route: `pf_beam`
- 状態: exp307 promotion gate FAILにより未実装・未実行のまま閉鎖
- 実装 / Kaggle package / push / run / inference / submission: すべて未承認・未実行
- CV / LB: なし

## 2026-07-21 設計確定

- contiguous finite run、lag 1--20、positive ACFだけを使う`n_eff`を固定した。
- leave-one-well-out median prior、`n_eff/(n_eff+50)`、log shrinkageを固定した。
- target-free triggerを満たさなければ0 HMM runで閉じる。
- trigger PASS時も最大1 variant x 773 wells、LightGBM/PF/Beam/boosterは0である。
- exp308/exp309との組合せ、パラメータgrid、inference、submissionは範囲外とした。

## 再現性メモ

- RNGなし。row連続run、lag、LOO median、式、thresholdを事前固定する。
- exp307 scale audit、ACF/n_eff、prior/shrunk sigma、trigger、prediction、metricsのcontent SHAを将来保存する。
- parent/controlは再実行しない。

## 次のアクション

exp307がPASSした場合だけ、別途ユーザー承認を得てsupport audit実装を検討する。現時点ではコードを実装せず、Notebookはfail-closedとする。

## 2026-07-22 dependency close

exp307 v2は全promotion gateをFAILした。必須parent条件が成立しないためtarget-free triggerも評価せず、実装、HMM、inference、submissionなしで設計branchを閉じる。
