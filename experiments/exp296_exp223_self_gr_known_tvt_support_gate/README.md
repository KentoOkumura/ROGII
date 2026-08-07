# exp296_exp223_self_gr_known_tvt_support_gate

## 状態

- ルート: `ensemble`
- 状態: Kaggle CPU version 3完了・performance guard FAIL・branch closed
- CV / LB: `12.159749140` / 未提出
- 作成日: 2026-07-19
- 親実験: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- 参照negative実験: `exp225_state_known_tvt_self_gr_hmm_emission`

## 仮説

exp223のsame-well descriptor motifはvisible prefixに存在したTVT stateにだけ経験的根拠を持つ。candidate stateがknown `TVT_input` range外のときself-GR boostを完全neutralにすれば、support内の有効なboostを保ったままwrong-depth attractionとworst-well regressionを減らせると仮定した。

## 単一変更

exp223 best `hmm_selfgr_boost_only_a070_c100`を完全固定し、full-grid centering・scaling・positive clip後のself-GR boostへ次のmaskだけを掛けた。

```text
state_supported[j] = known_tvt_min <= grid[j] <= known_tvt_max
exp296_boost[row, j] = exp223_boost[row, j] if state_supported[j] else 0.0
```

- `known_tvt_min/max`はfinite visible-prefix `TVT_input`全行から作る。
- boundaryはinclusive、padding 0。
- support外self-GR contributionはexact 0。
- support内boostはexp223とexact parity。
- base Type Well HMMはsupport外でも通常どおり動き、最終予測をknown rangeへclipしない。
- final predictionやtrue TVTをgate入力に使わない。

数式、適用順序、不変条件、hard gateの正は[support_gate_contract.md](support_gate_contract.md)とsteeringとする。

## 実行規模

Kaggle private CPU version 3で3,783,989 rows / 773 wellsを完走した。新variant 1本、773 HMM well-runs、LightGBM config / trained fold / booster `0 / 0 / 0`、GPU 0、parent/control再実行0。runtimeは16,667.265秒（4.630時間）。

## 検証方針

保存済みexp223 controlをdecompressed SHAとrow identityで固定し、official unknown suffix全rowをtruth-late join後に比較する。stable SHA256 well hashの5 reporting folds、true-TVT inside/outside known range、distance、hidden-like 2面、by-well、step deltaを読む。technicalとperformanceの事前登録hard gateを全必須とし、1項目でもFAILなら救済せず閉じる。

## 結果

| メトリック | exp223 control | exp296 | delta |
| --- | ---: | ---: | ---: |
| pooled RMSE | 11.349943 | 12.159749 | +0.809806 |
| pooled MAE | 6.471269 | 7.041758 | +0.570490 |
| within 10 ft | 0.794841 | 0.767778 | -0.027063 |
| true TVT inside known range RMSE | 10.044092 | 9.472290 | -0.571802 |
| true TVT outside known range RMSE | 13.164897 | 15.506322 | +2.341425 |
| 1000+ RMSE | 12.455457 | 13.352948 | +0.897491 |
| hidden-like spatial RMSE | 12.463402 | 13.574215 | +1.110813 |
| hidden-like typewell-purged RMSE | 12.266317 | 13.384951 | +1.118634 |

reporting foldはfold 3だけ`-0.258307 ft`改善し、残る4 foldsは`+0.530868`から`+2.289724 ft`悪化した。302 wellsが改善、471 wellsが悪化し、worst well `2364716c`は`+39.687791 ft`だった。

## 所見

technical guardは12/12 PASSしたため、strict maskの実装・比較自体は有効である。一方performance guardは2/10だけPASSし、事前登録した総合判定はFAILとなった。

true TVTがknown range内のrowでは改善したが、range外の1,459,531 rowsでそれ以上に悪化した。したがって「prefixで観測したTVT range外のcandidate stateにはself-GR evidenceの価値がない」という仮説は支持されない。same-well motifは、known TVT range外でも繰り返し形状や外挿的な位置合わせに有用な場合があり、candidate stateだけを根拠にhard zero化すると大きなmode errorを生む。

## 判定と次

契約どおり、このhard-gate branchを閉じる。padding、hole-aware/soft gate、alpha/clip/window/top-k/threshold救済、inference、submissionは行わない。

独立した次案としては、hard gateではなくtarget-freeなself-GR quality、posterior outside-support mass、known-range overlapをMLのadd-only risk featureとして読む既存低優先backlogだけを残す。exp296を直接救済するものではない。
