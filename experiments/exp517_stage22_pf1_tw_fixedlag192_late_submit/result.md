# exp517 stage 2-2 5 PF + fixed-lag 192 + tabular 再現結果

## 状態

修正版v2の学習、推論、LATE SUBMIT、公式scoringは終了した。postprocess OOF CVは`7.536732`、Publicは`6.778`で、stage 2-2掲載値`7.50 / 6.724`に近い。一方、Privateは`7.816`で掲載値`7.404`との差が`+0.412`あり、Privateまで含む完全再現ではない。train version 1はNotebookのkernelspec欠落で学習前に失敗した。旧推論v1は`1 PF direct`でstage 2-2契約を満たさない失敗履歴として保持する。2026-08-08、ユーザー判断により、この未再現判定を維持したまま実験を`completed`とした。

## 修正版v2契約

- PF: `pf_1 / pf_2 / pf_3 / r0_seed32 / r1_seed32` × `twGR`、32 seeds、fixed-lag 192。
- stage 2-2以後のStudent-t、tempering、ps-combo、anchor、emission、self/nbr、whole smoothingは無効。
- target: `TVT - last_known_tvt`。
- tabular: 公開Ravaghi base schema + 5 PF由来feature、3 LightGBM + 2 CatBoost × 5 GroupKFold、positive Ridge 5 folds。
- decode: `0.91 * Ridge + 0.09 * pf_1`、tau 85 fade、inference SG(17,3)。
- 実行量: scientific variant 1、base models 25、Ridge models 5、control rerun 0。

## v2 Kaggle train結果

| 指標 | 値 |
| --- | ---: |
| positive Ridge OOF RMSE | `7.531449` |
| public postprocess OOF RMSE | `7.536732` |
| SG(17,3) diagnostic OOF RMSE | `7.536165` |
| published stage 2-2 CV | `7.500000` |
| postprocess差 | `+0.036732` |

- kernel: `kentookumura/exp517-stage22-5pf-fl192-tab-train` version 2
- runtime: `14,603.468 s`、Tesla T4 x2
- coverage: 3,783,989 rows / 773 wells / 280 features
- PF: 5 banksすべて773 wells、合計PF runtime `4,145.890 s`
- models: LightGBM 15 + CatBoost 10 + positive Ridge 5。manifestに全model SHAを保存した。
- OOF decompressed-content SHA256: `31ce9041decf340de0f91a0d33de86bb68b35541e9d995158e17dcd6a3d695bc`

CVは掲載値と0.49%差まで一致したため、5 PF + tabular契約をhidden testで評価するgateを通過した。ただし掲載値`7.50`そのものとの数値完全一致は主張しない。inference version 1とsubmit-checkを完了し、LATE SUBMIT ref `55340618`は`COMPLETE`となった。

## v2 inference / submit-check

- kernel: `kentookumura/exp517-stage22-5pf-fl192-tab-infer` version 1、Tesla T4 x2、internet off
- public sample: 3 wells / 14,151 rows / 280 features
- 5 PF runtime: `55.618 s`
- duplicate ID 0 / missing 0 / finite / runtime sample order exact
- submission SHA256: `7a89df26198d04a7419c166a2f645f6b8a376d9d0924e4ffc3b7bffaa097ae45`
- candidate content SHA256: `661bd05214e9ae9bfac3ecfdd713fb0f97806556c83cbb1c40c5e765273d810b`
- `kaggle-submit-check`: FAIL 0 / WARN 0

## v1評価契約（失敗履歴）

最終公開v96 `pf_1` parameter、twGR、600 particles、32 seeds、GR likelihoodのみ、fixed-lag 192、direct smoothed-mean decodeを評価した。しかしstage 2-2掲載値は5 PF + tabular systemなので、このv1はstage 2-2再現ではない。

## Public commit runと提出前検査

- kernel: `kentookumura/exp517-stage22-pf1-tw-fl192-late-infer` version 1
- runtime: T4 x2 / 3 wells / 14,151 rows / PF `12.177876968 s`
- contract: `id,tvt`、duplicate 0、missing 0、finite、sample ID order exact
- submit-check: FAIL 0 / WARN 0
- submission SHA256: `ca9777cf782603f8cedfa4812b5762922015d8f43b10df345cee0cdb4ae2bb8d`
- candidate decompressed-content SHA256: `88aa41f1d9c510f649e6a9bd22ed9260ec26ff9866f16ce301b908b17afaa23d`

well数、runtime、SHAはpublic commit runの実測であり、hidden rerunの実測ではない。

## LATE SUBMIT 結果

| 対象 | Public | Private | 契約 |
| --- | ---: | ---: | --- |
| exp517 ref `55327703` | `7.825` | `9.689` | 1 PF direct decodeの本proxy |
| exp517 v2 ref `55340618` | `6.778` | `7.816` | 5 PF + fixed-lag 192 + 公開tabular stack |
| stage 2-2掲載値 | `6.724` | `7.404` | 5 PF + smooth + tabularの外部参照 |
| exp516 ref `55326266` | `10.056` | `8.552` | final-v96 pfA + anchor/emission + whole smoother |

submission ref `55327703`はKaggle CLI monitorとユーザーGUIの両方で`COMPLETE`を確認した。scoringは10分。stage 2-2外部参照との見かけの差はPublic `+1.101 ft`、Private `+2.285 ft`だが、入力数とtabular decodeが異なるため同条件の性能再現比較ではない。exp516比ではPublic `-2.231 ft`改善、Private `+1.137 ft`悪化だった。

修正版v2 ref `55340618`はKaggle CLIで`COMPLETE`、Public `6.778`、Private `7.816`を確認した。掲載値との差はPublic `+0.054 ft`、Private `+0.412 ft`。契約不一致v1からはPublic `-1.047 ft`、Private `-1.873 ft`改善した。監視はユーザー指示により22分時点で停止していたため、正確なscoring所要時間は不明である。

## v2再現判定

- CV: 掲載値との差`+0.036732`（`+0.49%`）で近似再現。
- Public: 掲載値との差`+0.054 ft`（`+0.80%`）で近似再現。
- Private: 掲載値との差`+0.412 ft`（`+5.56%`）で未再現。
- 結論: CVとPublicのscore regimeは再現したが、Privateを含むstage 2-2全体のLB再現は未達。名称変更や別結果への付け替えは行わない。

## v1判定

- hidden-compatibleな1 PF生成とlate scoringだけは完了した。
- 5 PFとtabularを省略したためstage 2-2手法契約gateはFAIL。技術的再現PASSという旧記録は撤回する。
- Public `7.825` / Private `9.689`をv2へ付け替えず、契約不一致のnegative evidenceとして保持する。

## Negative scope

v1のnegative resultは`(final-public pf_1 parameter, twGR, GR-only observation, standalone fixed-lag-192 mean decode, no fusion, hidden late rerun, 600×32, T4x2)`に限定する。v2が閉じる範囲は`(公開Notebookから復元した5 PF config, twGR, fixed-lag 192, 公開Ravaghi tabular stack, 現在のKaggle hidden rerun)`でPrivate掲載値を再現できなかったことまでとする。未公開の学習・実行差、当時のartifact差、PF family全体、final 6th-place systemは閉じない。

## 次のアクション

固定したv2で学習、推論、submit-check、LATE SUBMIT、公式scoringまで終了した。one-shot契約を消費済みのため、LB後調整や再提出は行わない。Private差の原因を未解決事項として残す。
