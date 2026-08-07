# exp444_acceleration_state_exact_hmm 結果

## 状態

独立仮説としてKaggle private CPU Stage 0A version 1
（id_no `129154702`）を完走した。exactness、normalization、leakage、RSSは
PASSしたが、fixed32/full runtime projectionが固定上限を超えたため
`stage0a_fail_closed`。Stage 0B/1、inference、submissionは未実施。
target-free technical preflightであり、科学結果、CV、LBはない。

## 仮説

3値acceleration状態がrate trendを複数行に持続させ、rateを各行で再発見する
exp209/exp441の遅れを減らす。

## 親からの変更

exp441のfull-support OU、exp209のposition/emission/prior/readoutを固定し、
3値のpersistent acceleration状態だけを追加した。exp441は構造参照と
negative contextであり、実行前提ではない。exp442は親でも先行条件でもない。

## 設定

- 構造参照・一要因control: exp441
- root比較対象: exp209
- Route: `pf_beam`
- acceleration: `[-0.0005,0,+0.0005]`
- transition: `0.08/0.84/0.08`
- control: 保存exp441、root reference: 保存exp209
- Stage 0A/0B: 4 wells / fixed32 total 32
- Stage 1: 全PASS・別承認時だけ773 wells
- model / booster / PF / Beam / GPU: 0

当初のexp441/exp442先行条件は2026-07-30のユーザー判断で撤回した。
exp441 Stage 0 FAILをnegative contextに固定し、明示trend-memoryが
full-support OU単体の不足を回復できるかという独立組合せ仮説として扱う。

## 結果

| メトリック | 値 |
| --- | --- |
| 設計 | 独立仮説として更新済み |
| compact train / inference | 実装済み / fail-closed実装済み |
| 専用test | 14 passed |
| py_compile / Ruff / Jupytext | PASS / PASS / PASS |
| strict experiment / template validation | PASS / PASS |
| 正規train Notebook / Kaggle Stage 0A | 採用済み / version 1 COMPLETE |
| Stage 0A rows / wells | 21,962 / 4 |
| candidate HMM / elapsed | 746.353694 / 767.339096 sec |
| fixed32 runtime projection | 5,970.829552 sec（上限3,600、FAIL） |
| full runtime projection | 144,232.851372 sec（上限30,600、FAIL） |
| peak RSS | 2.282776 GB（PASS） |
| finite / normalization | 1.0 / 7.371881e-14（PASS） |
| OU parity / dense prediction error | 0.0 / 3.219611e-09（PASS） |
| freeze前forbidden read | 0（PASS） |
| Stage 0B eligible | false |
| CV / LB | - / - |

## 解釈

3状態acceleration transition、destination-acceleration-conditioned OU kernel、
因子化exact forward/backward、zero-acceleration exp441 parity、small-state
dense referenceが実装・静的検証できた。fixed4 selectorはmanifestの`well`列だけを
SHA順に並べ、truth/role/fold/episode/causeをStage 0Aで読まない。

数値contractとtarget-free実行順序は成立し、RSSも安全だった。一方、
state数3倍のfactorized exact HMMでもfixed32換算約99.5分、full換算約40.1時間で、
事前固定上限を大きく超える。Stage 0Bへ進む前のruntime gateが意図どおり働いた。
predictionの科学評価やwrong-trend persistence評価には進まない。

## 次

事前登録どおりexp444をterminal closeする。state数、span、transition、kernel、
runtime実装、parameter、gateのsame-branch救済は行わない。
Stage 0B/1、inference、submissionへ進まない。
