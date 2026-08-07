# exp360 結果

## 状態

Kaggle private CPU version 2（id_no `128366385`）でStage 0を完了した。
technical / scientific gateの両方がFAILし、事前規則どおり
`close_zncc_confidence_branch_without_rescue`としてbranchを閉じた。
model、prediction、inference、submissionは生成していない。

## 仮説

raw absolute Gaussian likelihoodよりも、raw finite horizontal GRと
`GR_typewell(TVT_geop + δ)` のZNCC surfaceの方が、通常matching `δ=0`の破綻を
scale/offsetに依存しにくく捉え、exp264 bad10 errorをtarget-freeに予告できる。

## 固定設定

- 親: `exp340_exp226_depth_alias_block_confidence_readout_on_exp264`
- 対象: 773 OOF wells / 5 folds / expected 7,787 blocks
- block: 512 rows
- shifts: `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`
- finite pairs: 32以上
- primary: `best_nonzero_minus_zero_zncc`
- control: exp280保存raw Gaussian + stable SHA256 shift-label permutation
- model / booster / inference / submission: 0

## 判定予約

- supported block coverage `>=0.98`、全773 wellsにsupported blockあり。
- Q4−Q1平均block RMSE `>=+0.50 ft`、中央値正、正方向4/5 folds以上。
- pooled row-weighted bad10 AUC `>=0.60`、AUC `>0.50`が4/5 folds以上。
- 1000+、hidden-like spatial、hidden-like typewell-purgedの全scopeで正方向。
- exp280 raw Gaussian analog比pooled AUC `>=+0.02`、改善4/5 folds以上。
- stable permutation比pooled AUC `>=+0.02`、real優位4/5 folds以上。

## Stage 0結果

| 指標 | 値 | gate |
| --- | ---: | --- |
| wells / expected blocks | 773 / 7,787 | PASS |
| supported wells | 772 / 773 | FAIL |
| supported blocks / coverage | 7,700 / 0.988828 | PASS (`>=0.98`) |
| freeze前truth access | 0 | PASS |
| pooled Q4−Q1 mean block RMSE | +0.107479 ft | FAIL (`>=+0.50`) |
| pooled Q4−Q1 median block RMSE | +0.085354 ft | PASS (`>0`) |
| 正方向RMSE fold | 4/5 | PASS |
| pooled row-weighted bad10 AUC | 0.505164 | FAIL (`>=0.60`) |
| bad10 AUC `>0.50` fold | 4/5 | PASS |
| 1000+ Q4−Q1 mean block RMSE | −0.169027 ft | FAIL |
| hidden-like spatial / typewell-purged | +0.276701 / +0.214993 ft | PASS |
| raw Gaussian pooled AUC | 0.549949 | reference |
| ZNCC − raw Gaussian pooled AUC | −0.044785 | FAIL (`>=+0.02`) |
| raw Gaussianより良いfold | 1/5 | FAIL (`>=4/5`) |
| permutation pooled AUC | 0.488520 | control |
| ZNCC − permutation pooled AUC | +0.016644 | FAIL (`>=+0.02`) |
| permutationより良いfold | 4/5 | PASS |

全shiftがinvalidだったwellは `896d15b9` で、全well support条件を満たさなかった。
ただしtechnical failureを除いても、primaryのpooled効果量、AUC、1000+、
raw Gaussian差、permutation差が複数FAILしており、scientific rejectionは変わらない。

## 再現性

- deterministic anchor: いいえ。
- seed policy: real path RNGなし、controlはstable SHA256 per well/block。
- kernel: `kentookumura/exp360-typewell-shift-zncc-readout-train` version 2、
  id_no `128366385`、runtime `125.393474 sec`。
- score / valid mask / feature / quantile content SHA:
  `1c16ae1c...bab7` / `bf975b3b...3377` / `642b93dc...6c2` /
  `8733750c...90df`。
- post-freeze readout decompressed SHA:
  `7d41349afdb4f954b7a73f672364c2b0b7416ae996a34210dbccb152396a5f58`。
- 13件の生成物SHAをローカル再計算し、manifest不一致0件を確認した。
- freeze前truth accessは0。model / prediction / submission SHAは非該当。

## 解釈

ZNCCの非ゼロshift優位はpermutationより僅かに強かったが、差は`+0.016644 AUC`に留まり、
exp280保存raw Gaussianより`-0.044785 AUC`悪かった。primaryのQ4−Q1平均誤差差も
`+0.107479 ft`で、1000+では符号が反転した。したがって、scale/offset不変なshape
matchingへ置き換えても、exp264のbad blockを安定して識別するconfidenceにはならない。

supporting familyや4 sentinelの結果は合否に使わない。exp340のclosed判断も維持する。

## 次

threshold、family、shift grid、pair/std条件、sentinel、supporting familyによる救済を行わず、
ZNCC confidence branchを閉じる。prediction変更、add-only ML feature化、inference、
submissionへ進まない。この結果だけを根拠とする同familyの救済backlogも追加しない。
