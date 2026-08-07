# 設計

## 仮説

K12/K16/K24が同じpathを予測する領域ではexp226 geometry familyはscale-stableで、K16のselector順位を
上げる根拠になりうる。逆にK間のlevel、H128 slope、segment boundary近傍jumpが大きい領域は不安定である。
ただしexp300の問いは「K16を選べていない場所」であるため、primary scoreの方向を
「instabilityが高いほどK16がunderselectedされる」に事前固定し、外部valid truthを見て反転しない。

この診断はselectorを改善するものではない。raw-testで生成できるK-scale情報がselection regretとfold-stableに
関連するかを判定するだけである。

## 実験範囲

- 対象実験: `exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264`
- Route: `ml_model`
- 親: corrected exp264 Stage C v6 hard selection
- 候補入力: exp302のK12/K16/K24 prediction
- evidence: exp300 selection-regret分解
- 変更する変数: なし。固定scoreの診断のみ
- 対象外: model fit、candidate generation、routing、correction、inference、submission

## 先行条件とキャンセル

次の4条件を全て満たすまで実装しない。

1. exp302 technical PASS。
2. exp302 candidate novelty PASS。
3. corrected parentでのexp276再検証が完了。
4. exp276 promotion guard FAIL。

exp302 novelty FAILなら入力仮説がなくなるため閉鎖する。exp276 PASSなら既存の固定risk familyでtail-risk課題が
解決するため、ユーザー再承認なしでは実装しない。

## 固定feature schema

予測を`p12, p16, p24`とする。row単位で次を作る。

| feature | 定義 |
| --- | --- |
| `level_spread_ft` | `max(p12,p16,p24)-min(p12,p16,p24)` |
| `level_std_ft` | 3予測のpopulation std |
| `k16_midpoint_deviation_ft` | `abs(p16-(p12+p24)/2)` |
| `outer_asymmetry_ft` | `abs(p12-p16)-abs(p24-p16)` |
| `direction_agreement` | `sign(p12-p16)==sign(p24-p16)` |
| `k_order_monotone` | K順に単調増加または単調減少 |

未知suffix先頭をoriginとする非重複H128 blockごとに、row位置を`[-1,1]`へ正規化して各pathのOLS slopeを求める。

| feature | 定義 |
| --- | --- |
| `slope_spread_ft` | 3 slopeのmax-min |
| `slope_std_ft` | 3 slopeのpopulation std |
| `k16_slope_midpoint_deviation_ft` | `abs(slope16-(slope12+slope24)/2)` |

`boundary_weighted_jump_spread_ft`は、K12/K16/K24が示す等row segment境界のいずれかから±8行以内で、
3 pathの一階差のpairwise絶対差の最大値とし、境界外は0とする。final short segmentも含める。

## Primary score

outer foldごとにouter-train rowsだけで、次の3成分をaverage-tie empirical percentileへ変換する。

1. `level_spread_ft`
2. `slope_spread_ft`
3. `boundary_weighted_jump_spread_ft`

row scoreは3 percentileの算術平均、H512 block scoreはrow scoreのp90とする。外挿値は`[0,1]`へclipする。
fold、scope、truthを見たweight変更、成分削除、方向反転は行わない。残りのfeatureはsecondary診断だけに使う。

## Labelと評価

exp293と同じoriginで非重複H512 blockを作る。truth join後にのみ、

```text
benefit_ft = RMSE(exp264 Stage C selected hard) - RMSE(K16)
positive   = benefit_ft >= 0.25
```

を計算する。primary metricはpooled H512 block ROC AUC。fold AUC、score quintileごとのpositive rateと
mean benefit、1000+、hidden-like spatial/typewell-purged、by-wellを併記する。

scientific PASSは次の全条件である。

1. pooled H512 AUC `>=0.65`。
2. 4/5 foldsでAUC`>0.5`。
3. top quintile / bottom quintile positive-rate lift `>=1.5x`。
4. top quintileとbottom quintileのmean benefit差`>=0.25 ft`。
5. 1000+とhidden-like 2面で、両classがあればAUC`>0.5`、単一classならtop quintile benefitがbottomより大きい。

## freeze順序とリーク防止

1. exp302 K12/K16/K24とexp264 Stage C v6のrequired SHA、row/fold/well identityを照合する。
2. pre-freeze allowlistだけでrow/H128 featureとH512 blockを生成する。
3. outer-train empirical percentile map、feature schema/value、primary score、block/scope manifestをSHA freezeする。
4. truth-before-freeze access countが0であることを確認する。
5. 別loaderでtrue suffix TVTを接続しlabel/benefit/readoutを計算する。

`well_id`はjoin/group keyにのみ使用し、featureには入れない。exp300のoracle結果は仮説の背景だけで、score構築には使わない。

## 判断規則

- PASS: 別expでK-scale featureをcorrected exp264 selectorへadd-onlyする設計根拠になる。exp303内では学習しない。
- FAIL: K-scale stabilityによるselectability枝を閉じる。feature family、H64/H256、±境界幅、weight、threshold、
  score方向の同一OOF救済を行わない。

## 再現性設計

### 2026-07-21 実装固定

- exp302 K12/K24は、保存前float64に対する`prediction_content_sha256`を固定freeze manifest SHA
  `bd80a4e...b6919`の宣言で検証し、保存gzipはdecompressed SHAで独立検証する。CSVの12桁保存値から
  保存前content SHAを再計算したとは扱わない。
- K12/K24はchunk streamingし、unique ID文字列を全件保持しない。K16はwell/row/suffix/pred/foldだけを
  読み、canonical well-row位置へ配置する。
- empirical percentileはouter-train referenceに対するmid empirical CDF
  `(count(<x) + 0.5 * count(==x)) / n`へ固定する。
- H128 slopeは設計済みのとおりblock内positionを`[-1, 1]`へ正規化したOLS slopeとし、final short blockも含む。
- segment boundaryはwell suffix長に対する`np.linspace(0, n, K+1)[1:-1]`のcontinuous edgeとし、
  zero-based suffix positionが±8行以内ならboundary対象にする。
- 1000+ subgroupは、H512 block全行が1000+となる`block min MD >= 1000 ft`へ固定する。
- 既存正規Notebook placeholderは上書きせず、別名compact self-contained source/Notebookを実装対象とする。
- Kaggle package、正規Notebook採用、push/runは実装承認に含めず、別途明示承認を必要とする。

- seed/RNG: primary readoutに乱数なし。
- empirical map: foldごとにouter-trainだけで計算し、average tiesを固定する。
- 並列: 初回`num_workers=1`、stable orderは`fold/well_id/row_idx/suffix_offset`。
- runtime: CPU、0 model、0 booster、internet disabled。
- SHA: exp302 predictions、exp264 candidate score、hidden-like assignment、feature schema/value、empirical map、
  block assignment、primary score、readoutを記録する。
- Kaggle bootstrap: 実装・pushが承認された場合だけ確認する。submission/model manifestは対象外。

## リスク

- 方向誤指定: 高instabilityがK16優位とは限らない。truthを見た反転は禁止し、FAILとして残す。
- block相関: row AUCをprimaryにせずH512 blockで評価する。
- CV/LB: diagnostic AUCはLB改善を保証しない。PASSしても別のstrict nested add-only実験が必要。
- 条件付き多重性: exp302 best variantを選ぶ影響があるため、exp303はK12/K16/K24固定で追加K探索をしない。
- artifact整合: corrected Stage C v6 candidate score SHA不一致なら停止し、代替surfaceへ切り替えない。
