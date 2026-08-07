# exp445_tvt_to_u_coordinate_parity_exact_hmm

## 状態

- ルート: `pf_beam`
- 状態: Kaggle private CPU Stage 0 v2完了、coordinate parity確認
- CV / Public LB / Private LB: なし
- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 比較参照: `exp438_u_state_fixed_lattice_exact_hmm`
- 作成日: 2026-07-29

## 仮説

exp209の位置状態を、モデルを変えずに`TVT`から`U=TVT+Z`へ再表現し、
離散HMMを含めて親と同じ結果になることを確認する。

親の固定TVT格子を`P_j`とし、candidateでは各rowのU状態値を

```text
U_t,j = P_j + Z_t
```

とする。したがって`U_t,j-Z_t=P_j`で、TVTとして見た格子、GR emission、
posterior index、TVT readoutは親と同じになる。

## 変更点

- state indexと確率モデルは変えず、position stateの表示値だけを
  `P_j`から`P_j+Z_t`へ変える。
- moving U gridのphysical edgeを明示し、index-space transitionは親と同じ
  `r_current*delta_MD-delta_Z`に保つ。
- 性能variant、truth-based evaluation、inference、submissionは追加しない。

## exp438との違い

exp438は`P_j+Z_last`というabsolute U格子を全rowで固定したため、
TVTとして見た格子が`P_j+Z_last-Z_t`へ移動した。exp445は
`P_j+Z_t`とrowごとに既知量だけ平行移動し、TVT格子を常に`P_j`へ保つ。

そのためcandidateのindex-space position meanも親と同じ
`r_current*delta_MD-delta_Z`である。これはmoving U grid上では
`U_t,k-U_(t-1),j=(P_k-P_j)+delta_Z`となるためである。

## 検証方針

- synthetic variable-Z / constant-Zでcoordinate、physical edge、emission、
  prior、brute-force posterior parityを確認する。
- real fixed32でcandidate 32 + paired parent 32 = 64 HMM well-runsを比較する。
- suffix truth、fold、role、episode、errorは読まない。
- emission / transition / likelihood / position-rate posterior /
  TVT mean-std / `E[U]-Z` / SHAをAND gateで判定する。
- CV、RMSE改善、LB、promotionは判定しない。
- PASSならcoordinate parity確認として完了し、FAILならtechnical failureとして
  parameter rescueなしで止める。

## 実行入口

- 学習notebook:
  `exp445_tvt_to_u_coordinate_parity_exact_hmm_train.ipynb`
- 推論notebook:
  `exp445_tvt_to_u_coordinate_parity_exact_hmm_inference.ipynb`
- compact実装候補:
  `exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_train.ipynb`
- compact inference禁止guard:
  `exp445_tvt_to_u_coordinate_parity_exact_hmm_compact_selfcontained_inference.ipynb`
- compact候補は2026-07-30の実行承認後に正規notebookへ採用済み。
- Kaggle kernel:
  `kentookumura/exp445-tvt-to-u-coordinate-parity-exact-hmm-train`
  version 2、id_no `129095337`。
- full OOF、inference、submissionは本設計の対象外。

## 結果

| 項目 | 状態 |
| --- | --- |
| 設計 | 確定 |
| compact実装候補 | 完了 |
| 専用test | 17 passed |
| exp438 / exp445関連test | 29 passed |
| Jupytext / py_compile / Ruff F821 / strict validation | PASS |
| repository全体test | 既存exp297/301/333/336/349の5 collection errorで停止 |
| 正規Notebook採用 | 完了 |
| parity run | v2 `coordinate_parity_verified`、16/16 gates PASS |
| HMM実行量 | candidate 32 + paired parent 32 = 64 |
| posterior / rate / log-likelihood最大差 | `0 / 0 / 0` |
| TVT mean/std最大差 | `1.819e-12 ft` |
| runtime / peak RSS | `1920.670秒 / 1.190 GiB` |
| CV / LB | 対象外 |

## 解釈上の注意

parity PASSは「Uが新しい情報を追加する」「予測が改善する」という意味ではない。
既知Zによる一対一変換を正しく実装すれば、TVT状態とrow-shifted U状態が同じ
HMMになることだけを確認する。exp438のfixed absolute-U仮説も再分類しない。

## 所見

- row-shifted U座標はfixed32の離散exact HMMでも親TVT座標と数値一致した。
- 改善候補ではないため、成功条件をRMSEではなく親との数値一致だけに限定した。
- paired parent rerunは保存predictionだけでは確認できないposterior、
  log-likelihood、transition/emission parityの監査目的に限る。

## 次

run flagは完了後に再ロックした。初回成功runだけではdeterministic anchorとせず、
独立rerunは別承認がある場合だけ行う。inference / submissionへは進まない。
