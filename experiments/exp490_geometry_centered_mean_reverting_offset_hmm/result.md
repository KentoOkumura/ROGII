# exp490_geometry_centered_mean_reverting_offset_hmm 結果

## 状態

Kaggle private CPU Stage 0 version 1は固定all-AND gateでfail-closedとなった。
その判定を維持したまま、ユーザーの明示overrideにより同一1 variantを
4 target-free shardでfull 773-well OOF評価した。pooled RMSEとpersistent誤差は
大幅改善したが、by-well p95とworst-well gateがFAILしたため、最終状態は
`stage_1_full_oof_failed_closed`。その判定を維持した明示LB監査として、現行test
3 wellsへの推論とsubmissionファイル検証まで完了した。submission ref
`55163886`は受理されたが、version 1のhidden再実行が未処理例外となり、LBは付かなかった。
その後、hidden-dynamic version 2を実装・Kaggle実行し、生成物検証まで完了した。
version 2はsubmission ref `55180208`としてhidden再実行を通過し、Public LB `9.680`を得た。

## 仮説

exp357の主な残存誤差は、GRにより誤ったresidual offset / rate basinへ入った後、
geometryへ戻る力が弱いために誤りを維持することから生じる。K16区間1つの
half-lifeでoffsetとrateを0へ平均回帰させると、このpersistent errorを減らせる。

## 設定

- 親: `exp357_exp226_huber_emission_independent_audit`
- 検証: Stage 0 fixed32機構確認後、別承認時のみ5-fold / 773-well full OOF
- メトリック: RMSE TVT、fold、scope、by-well tail、persistent episode
- variant: K16 segment-span half-life 1候補
- シード: 42（本体は乱数なし）

## 結果

| メトリック | 値 |
| --- | --- |
| full OOF RMSE | **8.480155 ft** |
| exp357親RMSE | 9.737195 ft |
| exp357からの改善 | **1.257040 ft** |
| exp226 finalからの改善 | **0.946954 ft** |
| full rows / wells | 3,783,989 / 773 |
| 改善fold | 4 / 5 |
| persistent episode SSE reduction | **41.4100%** |
| persistent episode count delta | **-59** |
| 改善well / 悪化well | 449 / 324 |
| by-well delta RMSE p95 | **+7.257814 ft（FAIL）** |
| worst-well delta RMSE | **+49.602560 ft（FAIL）** |
| Stage 1 | fail-closed（12 / 14 gate PASS） |
| Stage 0 | fail-closed（fixed32、CVではない） |
| Public LB | **9.680**（version 2、ref `55180208`） |
| Private LB | なし |

## current-test inference

Kaggle private CPU kernel
`kentookumura/exp490-geometry-mean-revert-offset-hmm-inference` version 1
（id_no `129323029`）で完了した。exp226の`PredictionResult.geop`をSHA固定して
再生成し、full OOFと同じscientific contractの平均回帰HMMを3 wellsへ適用した。

| 項目 | 値 |
| --- | ---: |
| rows / wells | 14,151 / 3 |
| technical gate | 13 / 13 PASS |
| submit-check | FAIL 0 / WARN 0 |
| HMM runtime | 207.104秒 |
| total runtime | 265.469秒 |
| peak RSS | 1.159 GiB |
| submission SHA | `3970e9ad6d89250e3946f48fa97ed89b6dfd05dd33767514502ca8ca7f3be6e5` |

sampleとheader・行数・ID順序が一致し、ID重複、欠損、NaN、Infはない。
Stage 1 fail-closeは保持している。

## competition submission

- ref: `55163886`
- submitted: `2026-08-01 13:59:07.600000 UTC`
- kernel: `kentookumura/exp490-geometry-mean-revert-offset-hmm-inference` version 1
- Kaggle API status: `COMPLETE`、Public scoreなし
- error: `Your notebook hit an unhandled error while rerunning your code...`

公開commitは正常だったが、推論sourceは実行開始時に公開sampleのSHA、14,151 rows、
3 wellsを固定assertしている。Kaggleのhidden testは行数・well構成が異なり得るため、
この公開test専用ガードがhidden再実行と非互換である。提出CSVの形式不良ではない。
この提出時点ではhidden対応版の実装・再提出は承認範囲に含めなかった。後日のversion 2
実装結果は次節のとおりで、Kaggle実行・再提出は引き続き行っていない。

## hidden-dynamic inference version 2

2026-08-02の修正承認により、物理モデルを変えずruntime test契約だけを修正した。
公開sample SHA / 14,151 rows / 3 wellsはaudit-only参照値として残し、合否条件から外した。
version 2は重いexp226 full fitより前に、mountされたsample、全horizontal、全typewellを
一括走査し、全well集合と全`TVT_input`欠損rowをsample IDへ完全照合する。そのruntime
rows / wellsをexp226 geometryとexp490 HMMの実行量・coverage gateに使う。

train notebook、4 shard、strict merge、OOF prediction、HMM式・係数は無変更で、
scientific contract SHAは引き続き
`6398bbac380d3eca3a6255681b22c44c26de268ce6d4fad9dd242c066f2b9a35`。
したがってCV `8.48015525957654`を含むtrain結果は変わらない。公開3 wells全件と
synthetic 2-well可変sampleのpreflight契約を検証済み。Kaggle version 2のpush/runと
再提出を分離した上で、version 2のpush/runだけを別承認で実施した。全18契約test、py_compile、Ruff F821、Jupytext
round-trip、strict experiment validationはPASS。実行したstrict packageは929,792 bytes、
26 cells、output 0で、bootstrap内config/source SHAはlocalと一致した。

Kaggle private CPU version 2（id_no `129323029`）は`COMPLETE`。14,151 rows / 3 wells、
runtime inventoryを含むtechnical gate 14 / 14 PASS、HMM `110.832150 sec`、total
`149.097627 sec`、peak RSS `1.155529 GiB`。submit-checkはFAIL 0 / WARN 0で、
sample ID内容・順序、unique、finiteも独立確認した。prediction / submissionはversion 1と
byte-identicalで、submission SHAは
`3970e9ad6d89250e3946f48fa97ed89b6dfd05dd33767514502ca8ca7f3be6e5`。
version 2はユーザーの明示承認後にsubmission ref `55180208`として提出した。
提出時刻は`2026-08-02 07:23:53.240000 UTC`、kernel versionは2、提出前checkは
FAIL 0 / WARN 0。Kaggle API `COMPLETE`、Public LBは`9.680`、Private LBは未公開で、
version 1のhidden rerun errorは解消した。一回限りの提出承認は使用済みとしてfalseへ
戻しており、追加提出は行わない。

| LB比較 | delta（exp490 - 比較先、低いほど良い） |
| --- | ---: |
| CV 8.480155 | +1.199845 ft |
| exp226 direct 9.837 | **-0.157 ft** |
| direct exact HMM 9.063 | +0.617 ft |
| direct self-GR HMM 9.318 | +0.362 ft |
| direct likelihood PF 8.797 | +0.883 ft |

平均回帰はexp226 geometry単体よりLBを小幅改善したが、他のdirect物理decoderには届かなかった。
pooled OOFの大幅改善がLBへ同程度には移らず、固定強度mean reversionを物理routeのLB anchorへ
昇格させる根拠にはならない。Stage 1のwell-tail FAILと整合的にfail-closeを維持する。

## Stage 1 full OOF

### RMSE

| fold | exp490 | exp357親 | exp490 - 親 |
| ---: | ---: | ---: | ---: |
| 0 | 8.935035 | 8.002739 | +0.932296 |
| 1 | 8.659383 | 9.618405 | -0.959022 |
| 2 | 8.922330 | 10.113222 | -1.190892 |
| 3 | 7.928528 | 11.045039 | -3.116511 |
| 4 | 7.913022 | 9.655770 | -1.742749 |
| pooled | **8.480155** | **9.737195** | **-1.257040** |

MAEは`5.243598 -> 4.606500 ft`、絶対誤差5 ft以内率は
`69.1514% -> 71.4174%`、10 ft以内率は`83.8703% -> 86.1523%`へ改善した。

### scopeと長期誤差

| 指標 | exp490 - exp357 |
| --- | ---: |
| MD 1000+ RMSE | -1.434059 ft |
| hidden-like spatial RMSE | -1.306581 ft |
| hidden-like typewell-purged RMSE | -1.267906 ft |
| persistent episode SSE | -41.4100% |
| recovery rate @256 | +0.036050 |
| recovery rate @512 | +0.025078 |

### well別tail

773 wells中449 wellsは改善、324 wellsは悪化した。well別delta RMSEの中央値は
`-0.057105 ft`だが、p90は`+3.059139 ft`、p95は`+7.257814 ft`、p99は
`+17.501421 ft`である。最悪well `389ae58f`は親`3.764936 ft`から
candidate `53.367497 ft`へ悪化した。この不均一性が最終fail-closeの理由である。

## Stage 0 gate

### PASSした主な機構指標

| 指標 | candidate / 差分 | 固定条件 | 判定 |
| --- | ---: | ---: | --- |
| persistent episode SSE reduction | 69.8934% | 5%以上 | PASS |
| persistent改善well | 13 / 16 | 10以上 | PASS |
| persistent改善fold | 5 / 5 | 4以上 | PASS |
| matched-control pooled RMSE | 4.409685 ft | 親4.871908 ftから悪化`<=0.02 ft` | PASS（-0.462223 ft） |
| persistent episode count delta | -4 | 0以下 | PASS |
| recovery rate delta @256 | +0.080000 | 0以上 | PASS |
| recovery rate delta @512 | +0.120000 | 0以上 | PASS |

### FAILした固定gate

| 指標 | 値 | 固定上限 | 判定 |
| --- | ---: | ---: | --- |
| full 773 runtime projection | 51,464.889秒 | 30,600秒 | FAIL |
| matched-control by-well delta RMSE p95 | +3.118472 ft | +0.25 ft | FAIL |

technical側ではruntime projection以外の12項目をPASSした。manifest、K16区間、
positive dMD / span、`rho`、segment累積half-life、zero-state identity、
posterior normalization、finite、truth-before-freeze、peak RSSはすべて正常だった。
mechanism側ではcontrol tail以外の6項目をPASSした。

persistent foldのcandidate-minus-parent RMSEはfold 0から順に
`-5.316599 / -12.777886 / -4.596951 / -1.518047 / -0.093476 ft`で、
5/5 foldsが改善した。ただしmatched-controlのpooled平均改善と
by-well p95の大幅悪化が同時に生じており、安全な平均回帰ではない。

## 実装検証

- compact self-contained train source: 2,389行、9章
- 親exp357 compact train: 2,943行、12章
- exp490はStage 0 fixed32だけに絞るため章数は少ないが、入力、K16境界、
  Huber emission、exact forward-backward、truth-late readout、gate、生成物保存を
  notebook上で追える。
- `py_compile`: PASS
- `ruff --select F821`: PASS
- Jupytext変換 / `--test`: PASS
- exp490契約test: Stage 0、full、inference契約を合わせて15件PASS
- fixed32 manifest SHA:
  `fbbc62b7cb79e16a7fb436f3a9d11f8975e935ad2475a17e2dec4fd7b142e4d6`

## 参照値

| 参照 | RMSE |
| --- | ---: |
| exp357 Huber residual-offset HMM | 9.737195157482754 |
| exp281 Gaussian residual-offset HMM | 9.827419940583813 |
| exp226 final | 9.427109596582222 |
| exp226 geometry-only | 10.077950290784381 |

## 再現性

- deterministic anchor: いいえ
- seed policy: no RNG、固定well / row / segment / state順序
- Stage 0 kernel:
  `kentookumura/exp490-geometry-mean-revert-offset-hmm-train` version 1、
  id_no `129180511`
- full shard kernels: `exp490-mean-revert-full-shard0`--`shard3` version 1、
  id_no `129283179 / 129283181 / 129283176 / 129283180`
- strict merge kernel: `kentookumura/exp490-mean-revert-full-merge` version 1、
  id_no `129321382`
- executed bootstrap config SHA:
  `3c47bd5dc35fd68d4818fb25e49b8dd82170dc60e2551315e0b7a2a6a98cd315`
- scientific contract SHA:
  `221f6572bc1386475c87ca6db9eccad220ec5ec766e1aa56620611881ee0fbe0`
- decoder contract SHA:
  `ad75aa3190edd0bcee2f6ced088ef535317506eae8a5a660842c9181de2c91cf`
- prediction content SHA:
  `0098f0e9ee23e23d6a7f53cd63ae72bcbe3f546fd8c3b425672131560e2d6ca8`
- full executed config SHA:
  `71ac72507bda8bf6bd261b9dfe55d4dbebf51f2910fcfd009b9f97bc3086735d`
- full scientific contract SHA:
  `6398bbac380d3eca3a6255681b22c44c26de268ce6d4fad9dd242c066f2b9a35`
- full prediction SHA（gzip / 展開後）:
  `99030b33d493cc5f195f7d1a867f0d812a539143da9e1f59277e53779261b72c` /
  `e020e82e748a7836085657c4058070ff7853ed285639f2c2555cab721f9e9a07`
- inference kernel: `kentookumura/exp490-geometry-mean-revert-offset-hmm-inference`
  version 1、id_no `129323029`
- inference executed config SHA:
  `244b7041e661a27fc0fa031f59f0291b845ad62dc735c0b6f448c95b24b6e30a`
- inference prediction SHA（gzip / 展開後）:
  `413fa695ad32385f97c1d18a1947bbd7415687a274d1ecee03f02c22467e1cce` /
  `f5b7da9dc99387fef66a159a61d6e1e3c71368296f3b9cf075ec236bfa5845dc`
- submission SHA:
  `3970e9ad6d89250e3946f48fa97ed89b6dfd05dd33767514502ca8ca7f3be6e5`
- rerun result: favorable rerunは行わない

## 解釈

K16区間1つのhalf-lifeによるgeometry平均回帰は、full OOFでもpooled RMSE、
3つの重要scope、persistent episode SSE、episode数、recoveryを改善した。
「長く続く誤差を戻す」という物理仮説には強い支持がある。一方、正しい長期offset
まで強く0へ戻すwellがあり、449改善 / 324悪化という混合効果になった。
したがって固定強度の全well適用は不合格で、改善要素は適用強度を物理量から
決める遷移モデルへ分解して活かすべきである。

## 次

exp490そのものはfail-closeとする。version 1はhidden再実行ERRORだったが、修正済み
version 2はref `55180208`でPublic LB `9.680`まで完了した。Stage 1のtail FAILは保持し、
LBはhidden-dynamic動作と外部整合性の監査値として扱う。exp226単体からの小幅改善は
復元力の補助利用を支持する一方、direct物理候補としては競争力不足なので固定strengthの
救済調整や追加提出は行わない。
物理モデル側の次は保存済み
full OOFだけを使う0-HMM readoutで、悪化wellと改善wellのsegment span、観測GR情報量、
geometry不確実性、初期offset、suffix horizonを比較する。その結果を用いて、
平均回帰係数を固定値ではなく物理的な復元力・不確実性として定義する新実験へ進む。
