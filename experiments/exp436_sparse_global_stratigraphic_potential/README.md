# exp436_sparse_global_stratigraphic_potential

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU Stage 0 `stage0_fail_closed`
- CV / Public LB / Private LB: なし
- 作成日: 2026-07-29
- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- compact train / test: 実装済み
- 正規train notebook / package / run: 実施済み（kernel version 2）
- Stage 1 / Stage 2 / inference / submission: 未実施
- notebook: compact self-contained候補を正規train notebookへ採用済み

## 仮説

6つのformation contactごとにouter-trainの
`U_k=TVT_contact,k+Z_contact,k`を集め、各foldで1つの疎なglobal surfaceを解く。
target suffixは生formation列を読まず、最後の既知点における各surfaceとの差だけを使う。
これにより、exp226の局所donor mismatchを長距離積分せず、地層準位を保った保存的遷移を作る。

## 変更点

- `ANCC / ASTNU / ASTNL / EGFDU / EGFDL / BUDA`を6つの固定地層面とする。
- outer-trainの`Z-F_k=0` first crossingからcontact `U=TVT+Z`を作る。
- 1 global sparse surface / formation / fold、合計30 fieldsを解く。
- targetでformation列を読まず、各surfaceを64 ft control pointsでqueryする。
- anchorと全suffixでsupportを満たすformation集合を固定し、最低4面を要求する。
- primaryは固定集合のanchor差を等重み平均した保存場差。
- 行ごとのformation切替、local fit、fallback、selector、blendは使わない。

## 既存実験との差

- exp226: local donor rateを逐次積分しない。
- exp273: target known-prefixから固定2D gradientをfitしない。
- exp383: 1,043,436 windowsごとの6 local surface fitを行わない。
- exp381: crossing geometryのpositive evidenceだけを使い、contact-TVT RMSE
  `44.770101 ft`のFAILは覆さない。絶対datumはanchor差で消去する。

## 検証方針

- Fold: exp226と同じouter 5-fold group-safe split。
- Group: `well_id`。
- Score: 3,783,989 suffix rows / 773 wells。
- Control: 保存済みexp226 OOF、CV `9.427109596582213`、再生成なし。
- Stage 0: contact/node/edge/surface/query、leakage、support、resourceを監査する。
- Stage 1: 既知prefix最後512 ftのrolling-originでconstant-`U` nullを5%以上改善する。
- Stage 2: truth-late OOFでexp226を0.25 ft以上、4/5 folds、long/hidden/tailのAND gateで改善する。

## 実行入口

`exp436_sparse_global_stratigraphic_potential_compact_selfcontained_train.py`
にStage 0/1/2を実装し、正規train notebookへ採用した。Stage 0だけをKaggle CPUで
実行し、Stage 1/2のauthorizationはfalseのまま維持した。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

Kaggle private CPU version 2（id_no `129058940`）は`COMPLETE`したが、
BUDA source contactがfoldごとに`5 / 4 / 4 / 5 / 6` wellsしかなく、
固定最小32をFAILした。ほか5面は各fold 555–618 wellsで、fold 0の5面solveは成功。
6面contractが揃わないためqueryを実行せず、Stage 0でfail closedした。

## 所見

### リスク / 注意

- formation contactが少ないfold/面ではglobal surfaceが同定できない可能性がある。
- faultや非平行層を初回モデルでは表現しない。
- exp381で絶対contact-TVT移送は失敗しており、anchor差でも改善する保証はない。
- formation除外や重み変更で同じOOFを救済せず、必要なら別実験にする。

## 次

exp436は再実行せず閉じる。固定5面contractを検討するなら別実験・別承認にする。
