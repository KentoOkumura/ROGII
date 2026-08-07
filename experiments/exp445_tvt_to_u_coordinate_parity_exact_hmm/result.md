# exp445_tvt_to_u_coordinate_parity_exact_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 2を完了し、technical gate 16/16をPASSした。
判定は`coordinate_parity_verified`。CV、LB、inference、submissionは対象外。

## 仮説

親の固定TVT格子`P_j`をcandidateで`U_t,j=P_j+Z_t`と再ラベルし、
moving U gridの既知`delta_Z`を含めてedgeを表現すれば、離散state index上の
transition、emission、prior、posterior、TVT readoutはexp209と一致する。

## 設定

- 親: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 比較参照: `exp438_u_state_fixed_lattice_exact_hmm`
- Route: `pf_beam`
- 変更: state値の表示だけを`P_j -> P_j+Z_t`へ変える。
- 固定: grid index / step / band / phase、rate、transition probability、
  emission、prior、forward-backward、TVT readout。
- technical audit実績: fixed32、candidate 32 + parent 32 = 64 HMM runs。
- model / booster / PF / Beam / GPU: すべて0。
- CV / RMSE / LB / promotion: 対象外。

## 実行

- kernel:
  `kentookumura/exp445-tvt-to-u-coordinate-parity-exact-hmm-train`
- version / id_no: `2 / 129095337`
- runtime: private CPU、internet無効、1 worker、Numba thread 1
- candidate / paired parent / total HMM well-runs: `32 / 32 / 64`
- reporting fold / LightGBM / model / booster / PF / Beam / GPU: すべて0
- version 1はNumba初期化後の環境変数変更でHMM実行前に失敗し、
  科学contractを変えない最小修正後のversion 2で完了した。

## 結果

| 項目 | 値 | gate |
| --- | ---: | --- |
| technical gates | 16/16 | PASS |
| real log-likelihood max abs | 0 | PASS |
| smoothed position posterior max abs | 0 | PASS |
| smoothed rate posterior max abs | 0 | PASS |
| TVT mean/std max abs | 1.819e-12 ft | PASS |
| `E[U]-Z` readout max abs | 8.882e-16 ft | PASS |
| position kernel max abs | 2.220e-16 | PASS |
| physical-edge residual max abs | 1.110e-16 ft | PASS |
| brute-force max abs | 2.946e-08 | PASS |
| finite coverage | 1.0 | PASS |
| forbidden truth/fold/role/episode/error reads | 0 | PASS |
| runtime | 1,920.670秒 | report |
| peak RSS | 1.190 GiB | report |

## 再現性

- deterministic anchor: false（初回成功runのみ）
- manifest SHA:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`
- metrics SHA:
  `51786ac378eaf9366eeee62e3d9466c7584a0667fd9d32bc1d9fc7d690a3d783`
- prediction decompressed SHA:
  `88c642571023dc2560ad57a59580df74977e4fe021fb0c5faa41455acc7a240c`
- posterior ledger decompressed SHA:
  `a3fdf9ceae971f1df47dda862a344cd000977d4c4a472d3605633bca89c00898`
- transition/emission ledger decompressed SHA:
  `8c698a7d501cc94d9fca76680fce9ea0a1f35db7d84924a9ddcdb21b13872824`
- artifact readback SHA: 全PASS
- model / submission SHA: 非該当

## 解釈

exp209の固定TVT格子`P_j`とrow-shifted
`U_t,j=P_j+Z_t`は、fixed32の離散exact HMM上でも同じ確率モデル・posterior・
TVT readoutを与えることを確認した。これは座標再ラベルの検証であり、
予測改善でもexp438のfixed absolute-U latticeの再評価でもない。

## 次

run flagを再ロックし、full OOF、inference、submissionへ進まない。
deterministic anchor化が必要なら、別承認された独立rerunでinput /
posterior / prediction SHA一致を確認する。
