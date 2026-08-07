# exp492 結果

## 結論

Kaggle private CPU version 1で固定40 boosterのStage Cと科学readoutまで完了した。
hard primary RMSEは保存exp264の`8.652531956`から`8.639368546`へ
`0.013163410 ft`改善したが、改善foldは`3/5`に留まり、
by-well p95悪化`+0.381470357 ft`とworst-well悪化`+4.254514134 ft`も
上限を超えた。凍結scientific gateはFAILで、
decisionは`FAIL_CLOSE_FIXED12_HUBER_REPLACEMENT_SELECTOR`。
weight、threshold、domain、gate救済、downstream、推論、提出へ進めず閉じる。

## 仮説

Huber exact HMMを13本目として追加せず、元のGaussian `exact_hmm` semantic slotと
置換して12候補を維持すれば、exp392で見えたcandidate count増加由来のrerankingを
除いてselector適合性を評価できる。

## 事前根拠

| 指標 | 値 |
| --- | ---: |
| exp389 Huber candidate RMSE | 11.852741129500146 |
| Gaussian control RMSE | 11.938287234887435 |
| candidate gain | +0.08554610538728902 ft |
| improved folds | 5 / 5 |
| exp392 fixed13 hard RMSE | 8.769791681950748 |
| parent fixed12 hard RMSE | 8.652531955610227 |
| exp392 delta | +0.11725972634052084 ft |

## 凍結した設定

- 12候補ID・順序・domainを維持
- 4 changed / 8 unchanged
- exp264 88列feature schemaを維持
- 1 variant / 2 objectives / outer 5 x inner 4 / 40 CPU booster
- control再学習0 / GPU 0 / downstream 0 / inference 0 / submission 0

## Kaggle実行

- kernel:
  `kentookumura/exp492-huber-exact-hmm-full-replace-exp264-train`
- `id_no`: `129217774`
- version: `1`
- private CPU / internet disabled
- trained selector booster: `40/40`
- notebook elapsed at error: `6112.213 sec`
- parent/control再学習: `0`
- output archive: 科学gateとSHA回収が必要なためversion 1だけ取得

Stage C、technical gate、selector score guard、科学readoutは完了した。
その後のfeature importanceセルがlong-form CSVの列名を誤って`gain`と解釈し、
`KeyError: Column not found: gain`でNotebookのterminal statusはERRORになった。
canonical sourceでは`importance_type == "gain"`をfilterして`importance`列を
集計するよう修正済み。追加40 boosterのrerunは承認scope外であり、
科学gateも既にFAILしたため実行していない。

## CV結果

| 指標 | exp492 | 保存exp264 | delta |
| --- | ---: | ---: | ---: |
| pooled hard primary RMSE | 8.639368546 | 8.652531956 | -0.013163410 |
| near 0--250 RMSE delta | - | - | -0.012444018 |
| distance 1000+ RMSE delta | - | - | -0.014342934 |
| hidden-like最大delta | - | - | -0.096364940 |
| by-well p95 delta | - | - | +0.381470357 |
| worst-well delta (`d2f3b1ab`) | - | - | +4.254514134 |

fold RMSE / parent差:

- fold 0: `8.889139046` / `-0.102580979`
- fold 1: `8.418792015` / `-0.007842910`
- fold 2: `8.953215832` / `+0.052712491`
- fold 3: `8.464441826` / `+0.037968822`
- fold 4: `8.455201265` / `-0.044357407`

改善は3/5 foldsで、事前固定の4/5条件を満たさない。
Huber依存4候補のtop1は`937,102 / 3,783,989 = 24.764924%`。
report-only fixed fallbackは`8.238331546 -> 8.222215557`
（`-0.016115990 ft`）だった。

## 再現性

exp389 decompressed content SHA、exp264 schema/control score SHA、候補contract、
fold、sampling keyを固定した。回収できた主要SHA:

- exp389 decompressed:
  `f5d44d9d9ee380bb7ea408006030363efbe8fcdb3573cfa18031b2d31c617f90`
- feature schema:
  `b91ec1517a82641fe4d96f41c97872151f273a8bbfcb537284f91d47aacf1035`
- compact schema:
  `e3a677610899cb33bf58262f4cf02f650300c8c2207c46b53588d3418162ea74`
- scientific gate:
  `63f59978f4d28deb1044d0799d49bd461fc2c1719ca9eab45b4a6680af9d2ce1`

ERROR後のKaggle outputは大きいscore/model生成物を保持しなかったため、
model manifest SHAとouter-valid score SHAは未回収。初回runをdeterministic
anchorにはしない。

## 実装検証

- fixed12 candidate ID/order、primary 11、fixed fallback 7を維持。
- `exact_hmm`と依存3 formulaだけをfloat32で再計算し、8候補は完全parity。
- exp389 allowlist 6列、gzip decompressed SHA、post-read content SHAをfail-close。
- global key join後のouter fold再分割、truth-late、固定88/74 schema probeを実装。
- strict nested Stage Cは1 variant / 2 objectives / outer 5 x inner 4 /
  40 CPU booster、control再学習0に固定。
- pooled/fold/near/1000+/hidden-like/by-well科学gateとreport-only usage/fallbackを実装。
- 専用test 9件、構文check、Ruff F821、Jupytext round-trip、
  strict experiment validationを通過。
- version 1でtechnical / leakage / selector score guardは全PASS。

## 次

このbranchは閉じる。同一OOFでのweight、threshold、domain、gate救済や
追加rerunはしない。exp493 Student-t fixed12 replacementの独立結果を待ち、
両replacementが示すwell-tail不安定性を回避できる別のtarget-free continuous
risk readoutに進むかを判断する。
