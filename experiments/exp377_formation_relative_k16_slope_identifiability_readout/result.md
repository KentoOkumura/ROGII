# exp377_formation_relative_k16_slope_identifiability_readout 結果

## 仮説

formation-relative K16勾配はTVT直接勾配より坑井間の構造差を分離できる。

## 設定

- 親: exp226
- 検証: exp226 outer 5-fold / well group
- Primary: 6 formation-relative reconstructed rateの事前固定median
- メトリック: segment rate RMSE、累積path RMSE、scope別・fold別・well別差分
- シード: RNGなし
- 実行量: 1 diagnostic / 6 surfaces / 5 reporting folds / model・HMM・PF・booster各0

## 実行結果

| 項目 | 結果 |
| --- | --- |
| compact self-contained train候補 | 実装済み |
| fail-closed inference候補 | 実装済み |
| 正規train Notebook採用 | 済み |
| Kaggle kernel | `kentookumura/exp377-formation-relative-k16-slope-readout-train` v2 / id_no `128452991` |
| Kaggle状態 | `COMPLETE` |
| 実行時間 | 最終log `489.191 s` |
| 専用test | 8 passed（共通test込み20 passed） |
| Jupytext round-trip | PASS |
| py_compile / Ruff F821 | PASS |
| strict experiment validation | PASS |
| Stage 0 | **PASS**（low supportはreport-only） |
| Stage 1 | **FAIL**（7 checksすべてFAIL） |

### Stage 0 integrity

| 項目 | 実測 | 閾値 | 判定 |
| --- | ---: | ---: | --- |
| rows | 3,783,989 | 3,783,989 | PASS |
| wells | 773 | 773 | PASS |
| K16 segments | 12,368 | 12,368 | PASS |
| outer fold runs | 5 | 5 | PASS |
| primary coverage | 1.0000 | >= 0.98 | PASS |
| surface fallback fraction | 0.0000 | <= 0.05 | PASS |
| effective donors p05 | **2.5947** | >= 10 | warning（report-only） |
| valid truth reads | 0 | 0 | PASS |
| valid formation reads | 0 | 0 | PASS |
| source / valid overlap | 0 | 0 | PASS |

target-free bundle logical SHAは
`944af71f245e5e4615953c7d69fbbb3f22e48757cf63d8474e16d0398a683e5a`で
v1と一致した。effective-donor check以外のblocking checkがすべてPASSしたため、
truthをlate joinしてStage 1を評価した。

### Stage 1 identifiability

| 項目 | direct baseline | median6 primary | 差 / 判定 |
| --- | ---: | ---: | --- |
| segment rate RMSE | 0.012301 | 0.038454 | relative gain `-2.126104` / FAIL |
| cumulative path RMSE | 16.100131 ft | 38.776238 ft | `+22.676107 ft`悪化 / FAIL |
| 改善fold | - | rate `0/5` / path `0/5` | FAIL |
| H512 | 3.184898 ft | 6.560901 ft | `+3.376003 ft`悪化 |
| by-well | - | 164改善 / 609悪化 | FAIL |
| well差分p95 | - | `+49.434562 ft` | FAIL |
| worst well | - | `a247e7cf`, `+408.044686 ft` | FAIL |

個別formationのpooled path RMSEもANCC `40.355628`、ASTNU `39.955468`、
ASTNL `39.557385`、EGFDU `39.356186`、EGFDL `39.548626`、
BUDA `39.022186 ft`で、6系列すべてがdirect `16.100131 ft`より悪化した。

## 再現性

- seed policy: no RNG、stable fold/well/segment/row order
- deterministic tie: distance、donor well id、donor segment id
- kernel version: v2 / id_no `128452991`
- push config/package/bootstrap SHA:
  `676e2c1ebfffe98ab38f8b847eec806861343d7cd626ccde99dc76306f76ded9`
- train source/package/bootstrap SHA:
  `cdf79f3e6c35a718ebe13cfca1bf70a5548a3274af17631d478bdca891fd0018`
- packaged Notebook SHA:
  `cd75b0571953d9bd901042b35b6664627c701b4ca0745099d0561be83df31b94`
- truth manifest logical SHA:
  `6cf3ad2ad64f075189e9f162258f9bd727a80be197df51addca8a5a66760c19e`
- segment actual logical SHA:
  `ef644b7c9e404c2925950c727ca7e8d4f1721361c475344a8c4ab25b7c00012c`
- input / feature content SHA: freeze manifestとSHA manifestへ保存
- gzip SHA: decompressed content SHAを主証拠にする
- model SHA / manifest SHA: fitted model 0のため非該当
- submission SHA: inference/submission禁止のため非該当
- deterministic anchor: 成功rerun一致未確認のためfalse

## 解釈

実装とKaggle実行は技術的に完走し、v2ではcontrolled comparisonまで到達した。
K16・近傍50・bandwidth 500 ft・ridge 1を固定したまま、地層面からの相対勾配へ
変換すると、segment rateと累積pathの両方が全foldで悪化した。地層面を戻す分解は、
坑井間差を除くより大きな推定誤差を加えたと解釈する。

`clean273`はこのrepoでは特徴allowlistであり行scopeではないため、target由来の架空集合を作らずpooled契約の別名として明記した。この定義を変える場合はKaggle実行前に設計変更として扱う。

## 次

固定方式を不支持としてbranchを閉じる。surface / kernel / scopeのposthoc救済、
exp378 / exp379 / exp380 / exp382、inference、submissionへ進まない。
