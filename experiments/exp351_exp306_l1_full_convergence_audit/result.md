# exp351_exp306_l1_full_convergence_audit 結果

## 状態

Kaggle CPU full audit version 1完了。固定all-series gateをFAILし、parameter救済なしでtechnical negativeとして閉じた。

## 仮説

exp306 Stage 0で128/128 seriesをtechnical PASSした固定L1設定は、全773 wells / 1,546 seriesでも設定変更なしに全件収束できる。

## 設定

- 親: `exp306_robust_rts_l1_convergence_calibration_audit`
- Route: `pf_beam`
- kernel: `kentookumura/exp351-exp306-l1-full-convergence-audit-train`
- version / id_no: `1` / `128354027`
- branch: `l1_iter2000_rho1_tol1e4`のみ
- L1: max ADMM 2000、rho 1、abs/rel tolerance `1e-4`、lambda式固定
- 実行量: 773 wells / 1,546 series-runs
- model / LightGBM / fold / HMM / PF / Beam / booster / control再実行 / GPU: すべて0
- scientific score / prediction / submission: なし / なし / なし

## 結果

`1,537/1,546` series（99.4179%）がconvergence/technical PASSしたが、horizontal 9 seriesがmax iteration 2000に達して未収束となった。typewellは`773/773` PASS、horizontalは`764/773` PASS。固定基準は`1,546/1,546`のため、`full_technical_fail_closed`と判定した。

未収束horizontal:

| well_id | rows | iterations |
| --- | ---: | ---: |
| `5138a660` | 4,829 | 2,000 |
| `53f23031` | 10,200 | 2,000 |
| `591cc951` | 6,669 | 2,000 |
| `5def1ce5` | 7,997 | 2,000 |
| `81bf5923` | 6,386 | 2,000 |
| `ae069086` | 6,741 | 2,000 |
| `b37fd114` | 9,786 | 2,000 |
| `c59b6c4a` | 7,460 | 2,000 |
| `d924e971` | 5,308 | 2,000 |

PASSしたgate:

- 親version 1のcontract/gate/summary/sample/input/output/status/parity SHA。
- raw identity 773 wells。
- status coverage 1,546 series、duplicate 0。
- finite input/output、length/order identity。
- silent fallback 0、error 0。
- 親64-well input/output/statusと8-well output/status/iteration exact SHA parity。
- runtime 329.250秒 <= 30,600秒。
- truth/scientific score、prediction、submissionなし。

FAILしたgateは`all_converged`と、それに従属する`all_technical_pass`だけだった。

## Runtime

- preparation: `12.829504 sec`
- solver: `222.476799 sec`
- audit total: `329.250058 sec`
- full gate limit: `30,600 sec`

## 再現性

- parent anchor: 全PASS。
- cross-run parity: 64-well sample、8-well parityともexact PASS。
- raw identity content SHA: `bbb687a1998092578583ce259309b49031d095bde57cbb26c0ab8808d2379b32`
- full input content / raw / decompressed SHA: `96ed2ebc...8c6b0` / `1852b9ab...a2624` / `fa71faa3...092c3`
- full output content / raw / decompressed SHA: `45d51d60...3a61` / `64b422a3...0eef` / `e005aec9...12d2`
- solver status content / raw / decompressed SHA: `d1968a08...3e66` / `d85af923...c155` / `b83702d3...281d`
- runtime: Python 3.12.13、NumPy 2.0.2、pandas 2.3.3、single worker、BLAS threads 1。

## 解釈

64-well Stage 0はL1 max2000のfull feasibilityを過大評価した。失敗はexception、NaN、fallback、順序、親再現性、runtimeではなく、全データ中9本の長いhorizontal seriesが固定停止条件を2000反復以内に満たさなかったことに限定される。

設計済みfailure policyに従い、iteration、tolerance、lambda、rho、adaptive rho、solver gridで救済しない。technical eligibilityを得られなかったため、scientific score、exp304 selected SWT変更、HMM/PF/Beam、inference、submissionへ進めない。

## 次

exp351をtechnical negativeとして閉じる。本結果だけを根拠とするL1 solver救済backlogは追加しない。
