# 要件

## 依頼

exp486の候補をselector候補群の1つとして評価する新規実験を設計する。
バックログ、実験ディレクトリ、steeringを作成して設計を確定するが、実装、
Kaggle実行、推論、提出はまだ行わない。

## 仮説

exp486のabsolute geometry unaryは保存exp404に対してpooled RMSEを
`10.914522073 -> 9.726938029`へ`1.187584044 ft`改善し、4/5 foldsと
全事前scopeを改善した。一方、by-well delta p95 / worstが
`+10.069321492 / +44.021977054 ft`で単独昇格に失敗した。

corrected exp264 fixed12 selectorが、exp486のtarget-free prediction、
geometry residual、geometry log factor、ESS、resampling率と既存候補との
disagreementから有効な局所だけを識別できれば、pooled改善を利用しながら
well-tail悪化を抑えられる可能性がある。

## 制約

- Routeは`ensemble`とする。exp486 likelihood-PF候補とLightGBM selectorの
  両方がhard prediction生成に本質的に寄与する。
- selector parentはcorrected
  `exp264_exp263_candidate_confidence_dual_selector` fixed12に固定する。
- 追加候補はexp486の
  `absolute_geometry_unary_sigma20_lambda050`だけとする。
- exp486 residual版、固定HMM-PF 50:50、absoluteと既存候補の新規pair/blendは
  候補に追加しない。
- 既存12候補の値・順序・式・domainを変更しない。absolute候補はprimary
  hard-select domainだけへ13本目として追加し、fixed fallback 7候補は不変。
- exp486 prediction / absolute mechanism ledgerはSHA固定済み保存生成物を使い、
  PFを再実行しない。
- exp486候補はexp226のgroup-safe OOF geometryとknown-prefixだけから生成済み。
  upstream foldは安全性監査だけに使い、selector featureにはしない。
- truth、error、oracle、scope、by-well result、promotion gate、Residual predictionを
  feature freeze前に読まない。
- outer 5 × inner 4、2 objectives、40 CPU selector boostersだけを将来の
  Stage A/C実装範囲とする。親selector/control再学習、GPU booster、downstream
  TVT、current-test候補生成、inference、submissionは0。
- LightGBM設定、sampling cap、fold、objective、candidate weight、usage threshold、
  scientific gateを変更しない。
- 同じOOFでのweight / threshold / domain / feature family / gate救済を禁止する。
- 実装とKaggle runは、それぞれ別のユーザー承認を必須とする。
- 再現性は`docs/06_reproducibility.md`に従う。

## 受け入れ基準

- exp496のsteering、design-only実験ディレクトリ、config、候補・特徴・出力契約、
  backlogが同じ凍結設計を表している。
- 13候補のID・順序、primary / fixed fallback domain、追加候補allowlist、
  upstream SHA、fold / leakage境界が機械可読に固定されている。
- 将来実行量が`1 variant / 2 objectives / outer 5 / inner 4 /
  40 CPU selector boosters`、親/control再学習0、GPU0と明記されている。
- selector score guard、利用率、parent fixed12対比のpooled/fold/scope/by-well
  AND gateが実装前に固定されている。
- exp486単独FAILを再分類せず、selectorがFAILした場合のterminal close条件と
  same-OOF rescue禁止が明記されている。
- deterministic submission anchorとは扱わず、入力decompressed SHA、feature
  schema/content SHA、model manifest、candidate score、summary、Kaggle versionを
  将来記録する契約になっている。
- canonical Notebookはplaceholderのままで、実装コード、package、Kaggle run、
  inference、submissionが作成・実行されていない。
