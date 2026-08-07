# exp252_pf_seed_medoid_selectability_audit 結果

## 状態

Kaggle CPU train v1完了。candidate内likelihood順位付けは部分支持、bank gateは不採用。
inference / submissionは行っていない。

## 実行

- Kernel: `kentookumura/exp252-pf-seed-medoid-selectability-audit-train` version 1
- Kaggle ID: `127304849`
- rows / wells: 3,783,989 / 773
- runtime summary: 86.053秒
- model config / fold / booster / PF replay: 0 / 0 / 0 / 0
- exp243 row candidates、cluster manifest / summary、PF diagnosticsの期待SHA: 全件一致

## Oracle scope再確認

| scope | base8 oracle RMSE | base8 + K8 oracle RMSE | delta | useful units |
| --- | ---: | ---: | ---: | ---: |
| row | 4.564605 | 3.216218 | -1.348387 | 1,666,895 / 3,783,989 |
| block 128 | 4.805040 | 3.399936 | -1.405104 | 13,186 / 29,948 |
| block 256 | 4.883135 | 3.511059 | -1.372076 | 6,539 / 15,174 |
| block 512 | 5.036480 | 3.719798 | -1.316683 | 3,274 / 7,787 |
| whole-well | 6.592426 | 5.499587 | -1.092839 | 374 / 773 |

K8 bankには全scopeでoracle headroomがある。以降は、このheadroomを事前固定したtarget-free
scoreが識別できたかをreal-vs-shuffledで判定した。

## Bank selectability

bank scoreは弱い。最良の`resampling_rate`でも5 scope平均AUC 0.541785、shuffled比
+0.040204、whole-well AUC 0.560593だった。`log_likelihood_std`は平均0.519930、
whole-well 0.517933。`k8_max_nearest_base_disagreement`はwhole-wellで0.470769となり、
shuffled 0.478402も下回った。

374 useful wellsのうちK8 bankを使うべきwellを、base8 fallbackに必要な精度で識別できる
固定gateは得られなかった。

## Candidate selectability

K8 medoid内ではlikelihood系scoreに一貫したsignalがあった。

| score | 5 scope平均AUC | shuffledとの差 | whole-well AUC | whole-well shuffled |
| --- | ---: | ---: | ---: | ---: |
| cluster likelihood mass | 0.574731 | +0.061858 | 0.675214 | 0.557947 |
| medoid likelihood rank | 0.578665 | +0.076162 | 0.655102 | 0.517431 |
| medoid likelihood gap | 0.575143 | +0.072390 | 0.654235 | 0.508644 |

3 scoreはすべて5/5 scopeでshuffled AUCを上回った。target-free likelihood診断は、K8内で
どのmedoidが相対的に良いかをrankする材料としては有効と判断する。

ただし固定top1は実用選択規則にならない。whole-wellの`cluster_likelihood_mass` top1は
useful-medoid coverage 0.516043、union-best match 0.280749、best K8へのregret平均2.416045 ft、
p90 6.275659 ftだった。さらに全773 wellsではselected lossがbest base8より平均
+3.194947 ft悪い。likelihood rank / gap top1もcoverage 0.467914、regret平均2.972815 ft、
best base8比+3.751717 ftだった。

## 判断

- **部分支持**: K8 medoid内のlikelihood mass / rank / gapはcandidate-confidence材料に残す。
- **不採用**: bank gate、固定top1、direct medoid replacement、target-free score単独selector。
- **実施しない**: 現時点でのraw-test PF再生成、単独score selector、inference、submission。
- **後続**: likelihood系診断は既存`topk_path_confidence_features`またはfold-safe selectorの
  add-only候補へ統合する。bank gateが弱いため、独立した固定規則selectorは追加しない。

## Selector候補としての位置づけ

`cluster_likelihood_mass`、`medoid_likelihood_rank_score`、
`medoid_likelihood_gap_from_best`は、**K8 medoid内のcandidate-ranking特徴量としてselector候補に
なり得る**。ただし、この3値だけでK8を選ぶ固定selectorにはしない。

進める場合は、base8を常時fallbackに残す二段構成とする。

1. 別のraw-test-safe特徴を含むfold-safe bank gateで、K8 bankを使うか判定する。
2. K8を使う場合だけ、likelihood mass / rank / gapを既存candidate selectorへadd-only投入する。

outer-well OOFでbase8 / 親selectorを上回り、fold、near、hidden-like、worst-well guardと
raw-test feature parityを通過した場合だけ推論候補へ進める。exp252単体は、このselector構成を
支持したのではなく、3特徴を候補集合へ残す根拠を与えた段階である。

## 生成時間

生成コストは二つに分かれる。

- **保存済みK8候補からselectability scoreを読む処理**: exp252実測で全3,783,989 rows /
  773 wellsが86.053秒。
- **raw入力から128-seed PFとmedoid候補を生成する処理**: exp243 v3実測で全3,783,989 rows /
  773 wellsが37,067.406秒（約10時間18分）。500 particles × 128 seeds / wellとK=3/5/8
  clusteringを含む。

exp243実測は平均47.953秒/well。公式資料のhidden test約200 wellsを同程度の長さと仮定した
単純比例は9,590.5秒、約2時間40分となる。これはraw-testでの実測ではなく、well数比例の
参考値である。K8-only化でK3/K5 clusteringは省けるが、支配項の128-seed PFは残るため、
短縮幅は未測定。selector modelの学習・推論時間も未測定で、この見積もりには含めない。
