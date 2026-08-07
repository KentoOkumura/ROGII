# exp390_parallel_strip_surface_registration_readout 結果

## 仮説

近接horizontal wellをquery-centricなparallel-strip座標へ登録し、同じalong-track位置の
outer-train `S=TVT+Z`をcross-track方向へtwo-sided補間すると、genericなwell-level KNNや
XY donor fieldより安全な物理TVT pathを作れる。

## 設定

- 親・control: exp226保存済みouter-5-fold OOF
- Route: `pf_beam`
- candidate: `parallel_strip_two_sided_fallback_exp226` 1本
- 検証: Stage 0 target-free / Stage 1 prefix rolling-origin / Stage 2 truth-late suffix
- メトリック: RMSE、fold、distance/hidden-like、by-well、report-only oracle
- seed: RNGなし、stable deterministic order

## 親との差分

exp226のouter 5-fold、保存OOF、score rows、fallback predictionは固定し、
query-centricな`(s,n)` registrationとsame-s two-sided `S=TVT+Z`補間だけを追加する。
exp226 controlは再生成せず、GR、Formation、fitted model、HMM、PF、Beamは使わない。

## 結果

compact self-contained train候補、fail-closed inference候補、専用contract testを
実装し、正規train Notebookを採用した。Kaggle version 1は入力resolverが3件の
`test/`を選択して本体計算前に停止したため、CVやStage 0 scientific resultには
数えない。resolver修正後のversion 2は16-well Stage 0を完了したが、
two-sided supportの3 gateをFAILし、`stage0_fail_closed`で終了した。

| メトリック | 値 |
| --- | --- |
| Stage 0 | FAIL |
| processed wells / rows | 16 / 73,586 |
| two-sided row coverage | 0.0（基準0.50以上） |
| two-sided well coverage | 0.0（基準0.75以上） |
| eligible-node donor p05 | 0.0（基準4以上） |
| query with eligible pair | 8 / 16 |
| max eligible pair per query | 2 |
| pair angle p95 | 1.769352°（PASS） |
| pair overlap p05 | 0.897013（PASS） |
| fallback finite coverage | 1.0（PASS） |
| forbidden reads / role overlap | 0 / 0（PASS） |
| runtime / projected full runtime | 60.401419秒 / 2,918.143546秒 |
| projected peak RSS | 0.657509 GB |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |

## 合格条件

- Stage 0: two-sided row coverage`>=0.50`、well coverage`>=0.75`、
  donor p05`>=4`、leakage/read 0、16-well resource gate PASS。
- Stage 1: vertical-only anchor比`>=0.25 ft`、4/5 folds、real donor-sが
  circular controlより`>=0.10 ft`良い。
- Stage 2 scientific: exp226比pooled`>=0.25 ft`、4/5 folds、
  eligible rows`>=0.50 ft`、1000+`>=0.25 ft`、hidden-like非悪化。
- Stage 2 promotion: improved-or-equal wellsが過半、by-well delta p95`<=0`、
  worst-well delta`<=+0.25 ft`。

## 再現性

- deterministic anchor: 未確立
- seed policy: RNGなし、不変keyによるstable ordering
- kernel: `kentookumura/exp390-parallel-strip-registration-train`
- kernel version: version 1 `ERROR`、version 2 `COMPLETE` /
  numeric id `128480051`
- runtime package bootstrap / config / source SHA:
  `c562826...52a01` / `2b4601...110a` / `3f9dd8...b1c6`
- SHA manifest:
  `1f8bc775423dca2a75690a97899f5f5f2aec35602948bebff31746ddae5fc825`
- geometry / pair / node / fit logical SHA:
  `bb7c2d...ba296` / `c75733...dd43` / `25d2b0...daaa` /
  `7ab50d...fb5e`
- model SHA / manifest SHA: fitted modelなし
- submission SHA: inference/submission禁止のため対象外
- rerun result: version 1は入力resolver修正対象、version 2でscientific Stage 0完了
- 実装検証: 専用test `11 passed`、py_compile、Jupytext round-trip、
  strict experiment validation PASS

## 解釈

角度と投影overlapは固定gateを満たしたため、parallel well自体は見つかる。一方で
16 wells中のeligible pairは合計10、queryあたり最大2で、4 donorかつ正負両側という
補間に必要なsupportを満たすnodeは0だった。したがって失敗原因はfit品質ではなく、
事前設計したcross-track補間を成立させる局所donor密度の不足である。
thresholdやone-sided外挿を救済すると仮説と安全条件が変わるため行わない。

## 次

exp390はStage 0 FAILで閉じ、773-well full run、Stage 1 / 2、inference、
submissionへ進まない。parallel-strip系を再検討する場合は、全773 wellsを対象とする
0-fit target-free support censusを別アイデアとして先行し、非退化supportの証拠を求める。
