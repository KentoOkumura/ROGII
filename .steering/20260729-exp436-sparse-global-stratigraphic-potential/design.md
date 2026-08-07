# 設計

## 1. 地層面別ポテンシャル

formation indexを

```text
K = [ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA]
```

に固定する。各outer-train horizontal well、各`k`について、MD昇順で
`Z - F_k = 0`となる最初のexact zeroまたは符号交差を取り、隣接行間を線形補間する。
接触点で真TVTも同じ係数で補間し、

```text
u_wk = TVT_contact,wk + Z_contact,wk
q_wk = (X_contact,wk, Y_contact,wk)
```

をformation `k`のsurface observationとする。multiple crossingのoracle選択、
順序によるpost-hoc修復、outer-valid contactの使用は禁止する。

各fold・各formationで、全outer-train contactを同時に使う1つの疎な正則化面
`U_hat_k(X,Y)`を解く。単一`P(X,Y)`へ全地層準位を混ぜない。

## 2. Sparse global surface

### Node / graph

- nodeは1 source well・1 formationにつき最大1 contact。
- stable keyは`(fold, formation_order, well_id, contact_md)`。
- 同じformation内で別wellの近傍8 nodes、最大4,000 ftを無向graph edgeにする。
- edge weightは`exp(-d_xy^2/(2*1000^2))`。
- formation間edgeは張らない。

### Objective

formation `k`ごとに次をHuber IRLS + sparse LSQRで解く。

```text
sum_i huber((p_i - u_i) / s_k)
+ 1.0 * sum_(i,j) w_ij * ((p_i - p_j) / s_k)^2
+ 0.05 * sum_i ((L_k p)_i / s_k)^2
+ 1e-8 * sum_i (p_i / s_k)^2
```

- `s_k`はouter-train `u_i`のMAD、floor=`1e-6`。
- Huber delta=`1.345`、初期L2 + IRLS更新5回。
- LSQR `atol=btol=1e-6`、最大2,000 iterations、accepted `istop in {1,2}`。
- worker/BLAS threadは1、node/edge/reduction順を固定する。
- 6 surfaces ×5 folds=`30` field fits、内部sparse solve最大180。
- formationごとのfinite source wellsが32未満、graph/solver不成立ならfail-closeする。

これはqueryごとのfitではなく、formationごとにouter-train全wellから面を1回だけ解く。

## 3. Target queryと遷移

- target raw formation columnsは読まない。
- 最後の既知点、64 ft MD control points、suffix終端で各`U_hat_k`をqueryする。
- 各formationで別source well最大16、最低8、距離4,000 ft以内のsolved nodesを使う。
- bandwidthは選択した最遠unique well距離を`[500, 4000] ft`へclipし、
  Gaussian weighted meanで評価する。local plane / local solveは禁止。
- formation `k`はanchorと全control pointsでsupportを満たす場合だけwellの固定集合
  `K_w`へ入れる。`|K_w|>=4`を必須とする。
- `K_w`はsuffix全体で固定し、行ごとにformationを選択・脱落させない。

primary conservative field differenceは、

```text
delta_U_bar(t)
  = mean_{k in K_w}[U_hat_k(X_t,Y_t) - U_hat_k(X_a,Y_a)]
TVT_pred(t)
  = TVT_input(a) + delta_U_bar(t) - (Z_t - Z_a)
```

とする。固定等重み平均は各`U_k`の平均ポテンシャル差なので保存的である。
6つのsingle-formation pathはmechanism report-onlyとし、promotion候補、重み選択、
well/row selectorに使わない。

## 4. Foldとleakage

- exp226と同じouter 5-fold well split、773 wells、3,783,989 suffix score rows。
- surface contact、graph、solverはouter-train wellだけから作る。
- outer-validからpre-freezeで読めるのは`MD/X/Y/Z/TVT_input`のみ。
- outer-validの真formation/TVTはsurface/query/prediction SHA freeze後の監査だけで読む。
- Stage 1 prefix rolling-originはouter-validの既知prefix内だけで行う。
- 公開3 test wellsはsurface、support、gate、weight選択に使わない。

## 5. 検証段階

### Stage 0: target-free integrity/resource

全foldのcontact/node/edge/component census、fold 0の6 surfaces、固定16 target wellsの
queryを実測する。

hard gate:

- expected wells/rows/folds=`773 / 3,783,989 / 5`。
- outer-valid source contact/node/edge overlap=`0`。
- target GR/raw formation/suffix truth reads=`0 / 0 / 0`。
- duplicate contact/node/edge key=`0`、finite source coverage=`1.0`。
- formationごとのsource wells`>=32`、surface solve全PASS。
- primary finite row coverage=`1.0`、`|K_w|` p05`>=4`。
- query unique source wells p05`>=8`。
- full runtime projection`<=21,600 sec`、peak RSS`<=16 GB`。

FAILならStage 1/2へ進まず、formation除外、surface graph、regularization、
support、aggregationを同じ実験内で変更しない。

### Stage 1: prefix rolling-origin

Stage 0 PASS後、outer-valid suffix truthを読まず、既知prefixが1,024 ft以上あるwellの
最後512 ftをpseudo suffixとして隠す。primary field差をconstant-`U` nullと比較する。

- eligible well coverage`>=0.90`。
- pooled RMSEをnull比`>=5%`改善。
- improving folds`>=4/5`。
- 512 ft endpoint absolute error delta`<=0`。
- target-free bundle freeze前のsuffix truth read=`0`。

FAILならStage 2へ進まない。

### Stage 2: truth-late direct OOF

Stage 1 PASS後だけ保存済みexp226 OOFとsuffix truthをjoinする。

- candidate pooled RMSE`<=9.177109596582213`
  （exp226 `9.427109596582213`比`>=0.25 ft`改善）。
- improving folds`>=4/5`。
- suffix 1000+ gain`>=0.25 ft`。
- hidden-like spatial / typewell-purged delta各`<=0.0 ft`。
- suffix 0--250 delta`<=+0.05 ft`。
- by-well delta p95`<=0.0 ft`、worst`<=+0.25 ft`。
- exp226 prediction correlation`<=0.9995`。

exp263 CV`8.238331`はreport-only reference。全PASSでもinference、submission、
blend、fault拡張は別承認とする。

## 6. 既存実験との差

- exp226: local donor rateを逐次積分しない。各formationのglobal potential差を1回使う。
- exp273: target known-prefixから固定2D gradientをfit・外挿しない。
- exp383: 1,043,436 windowsごとに6 local surfacesをfitしない。source contactは
  1 well ×1 formation最大1点、global fieldは最大30。
- exp381: crossing geometryのpositive evidenceは参照するが、絶対contact-TVT
  RMSE `44.770101 ft`のFAILは覆さない。formation/well固有datumはanchor差で消去する。

## 7. 再現性と確定範囲

- RNGなし。fold、formation、well、contact、node、edge、query順を固定する。
- KD-tree tieを完全keyで再sortし、worker/BLAS threadは1。
- raw input、fold、role-read、contact、node/edge、solver、query support、
  rolling-origin、OOF predictionのschema/logical content SHAを保存する。
- 最初のrunはdeterministic anchorではなく、同一設定rerunのSHA一致後に再判定する。
- 1 primary physical candidate、6 formation report-only paths、5 folds。
- global fields=30、内部sparse solve最大180。
- model / booster / HMM / PF / Beam / GPU=`0 / 0 / 0 / 0 / 0 / 0`。
- 初回設計確定時点では実装、test、実行可能notebook、package、run、inference、
  submissionは0だった。2026-07-29の実装承認後、compact self-contained train候補と
  contract testを追加した。正規notebook、package、run、inference、submissionは0のまま。

## 8. 実装反映

- Jupytext percent形式のcompact self-contained train候補へ、Stage 0 census/preflight、
  Stage 1 prefix rolling-origin、Stage 2 truth-late direct OOFを実装した。
- Stage 0 / 1 / 2は独立承認フラグを持ち、未承認段階へ自動で進まない。
- 6 formation report-only pathは保存するが、固定support集合の等重みprimary以外を
  selection、promotion、weight調整へ使わない。
- exp226 OOFはpre-freezeに4 identity列だけを読み、truth/control列はprediction bundle
  のSHA freeze後だけ読む。
- 正規notebook placeholderは上書きせず、compact候補の採用は別判断とした。
