# exp505_exp490_tau500_fade_fixed13_on_exp413 結果

## 結論

Kaggle private CPU version 1でStage Cを完走した。technical checksは全PASSし、tau=500
fade fixed13のhard OOF RMSEは`8.243315437`でraw exp501 `8.264890209`を
`0.021574771 ft`改善した。4/5 folds、固定7 scope、fade利用条件も通過した。

一方、fixed12比のby-well tailはraw exp501からほぼ変わらなかった。p95縮小は
`0.000036536 ft < 0.10 ft`、worst縮小は`0.173168079 ft < 1.0 ft`で、事前固定した
2条件をともにFAILした。判定は
`FAIL_CLOSE_WITHOUT_STAGE_D_OR_SAME_OOF_RESCUE`。Stage D、GPU、inference、submissionは
実装・実行せず、このbranchを終端閉鎖する。

## 実行契約

- Route: `ensemble`
- Kernel: `kentookumura/exp505-exp490-tau500-fade-fixed13-on-exp413-train`
- Version / id_no: `1 / 129519165`
- Runtime: CPU、internet off、`7312.583 sec`
- 実行量: 1 variant × 2 objectives × outer 5 × inner 4 = 40 boosters
- exp264 / exp501 control再学習: 0
- HMM / PF / Beam再実行: 0
- Stage D GPU / inference / submission: 0 / 0 / 0

## OOF

| scope | exp505 RMSE | raw exp501 RMSE | exp505 - raw exp501 |
| --- | ---: | ---: | ---: |
| pooled | 8.243315437 | 8.264890209 | -0.021574771 |
| fold 0 | 8.416177539 | 8.466895073 | -0.050717534 |
| fold 1 | 8.295453153 | 8.278298760 | +0.017154393 |
| fold 2 | 8.030499522 | 8.044158353 | -0.013658831 |
| fold 3 | 8.232938395 | 8.243424839 | -0.010486444 |
| fold 4 | 8.236416367 | 8.285808879 | -0.049392512 |

- fixed12 hard OOF: `8.652531956`
- fixed7 fallback OOF: `8.238331546`
- nonworse folds vs raw exp501: `4 / 5`
- direct fade RMSE: `8.447032559794`。exp503期待値との差`2.06e-10 ft`でparity PASS。

## 固定scope

| scope | exp505 RMSE | raw exp501 RMSE | delta |
| --- | ---: | ---: | ---: |
| raw GR observed | 8.402335160 | 8.424034574 | -0.021699414 |
| raw GR missing | 7.890389517 | 7.911708409 | -0.021318892 |
| missing fraction high | 7.563848326 | 7.600876658 | -0.037028331 |
| distance 0--250 ft | 1.608630755 | 1.612337315 | -0.003706560 |
| distance 1000+ ft | 9.049264191 | 9.065295758 | -0.016031566 |
| hidden-like spatial | 8.823707964 | 8.872328848 | -0.048620884 |
| hidden-like typewell-purged | 8.740694950 | 8.806552858 | -0.065857908 |

固定7 scopeはすべてraw exp501を改善した。fade候補top1率はpooled `55.2414%`、fold別
`57.0105 / 54.0928 / 56.6601 / 56.9319 / 51.5069%`で、利用条件もPASSした。

## Tail gate

| 指標 | raw exp501 | exp505 | 縮小 | 必要量 | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| fixed12比 by-well p95 delta | +2.904593926 | +2.904557390 | 0.000036536 | 0.10 | FAIL |
| fixed12比 worst-well delta | +18.394664149 | +18.221496070 | 0.173168079 | 1.0 | FAIL |

fadeは平均、fold、scopeをわずかに改善したが、raw exp490を入れたexp501のwell-tailを
materialには変えなかった。候補をfadeへ置換するだけではselector bank内の高利用とrerankingが
残り、tail riskをexp413へ渡す前提を満たさない。

## Technical / leakage

- 3,783,989 rows / 773 wells / 13 candidates / compact77。
- exp490はfeature freeze前にexact 8列だけを読込。forbidden truth/error/role/fold/scope列の
  読込0、source fold利用0。
- raw gzip SHA、decompressed SHA、global key、suffix、`md_since` parityを全PASS。
- outer-valid wellのinner assignment混入なし。outer-train compactはinner OOF、outer-validは
  4 inner-model ensemble。
- 40 models、25 compact partitions、18,919,945 compact rows、49,191,857 score-long rowsが
  全て契約一致。
- fixed7 fallback error parity max abs: `0.0 ft`。

## SHA256

- feature schema: `f3811d0da796b1634c6ec28ef50d8d22aa4bbe443c4692b725f498b3c9dcf79b`
- model manifest: `e0523757dfb79c2082ba78aa582a5f5c8cce55fe247969e89e8d29979905362b`
- compact manifest: `49812f4c31d925b5630c41d5446173bf60ef21d962acfe9c3ea52a60ea5f9eb8`
- outer-valid candidate score: `c319fff5b8d675c170cf6d75260f05add64524eb2d8377b6edcc1963b26403ab`
- scientific gate: `edb3c7c933f8db2e102bdf35a90ca36634ccd280e4137b09f039a3e984d10f07`
- 完全Kaggle log: `e46310da31ff6dad4419406780c55ecb5e01def67964cdd58e67246880eff91d`

## 解釈と次

tau=500はexp503のsame-full-OOF 29 profile後に選ばれたため、本結果をclean independent CVとは
呼ばない。さらに、固定fadeはdirect predictionとselector OOFの平均を改善してもwell-tailを
materialに縮小しなかった。Stage C FAILをthreshold / tau / alpha / feature / gateで救済せず、
Stage Dへ進まない。

次の新規候補は、保存済みexp501/exp505 score・choice・by-well artifactだけで、fade利用55%にも
かかわらずtailがほぼ不変だった原因を固定bucketで分解する0-model readoutとする。これは
exp505を再評価するものではなく、独立仮説の根拠作りに限定してP4へ置く。
