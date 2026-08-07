# exp286_geop_hmm_sparse_addonly_candidate_on_exp264 結果

## 結論

`geop_hmm`を他候補と同じ情報を持つ13番目candidateとして追加したfull13 selectorは、
Stage B、Stage C、Stage Dのすべてで元のparent12 selectorよりpooled RMSEを改善した。

最終Stage Dでは、parent12 compact add-onlyの`8.4608112376`に対してfull13 compact add-onlyは
`8.4037839136`、delta `-0.0570273240 ft`だった。したがって「selectorへ追加すると全体RMSEが
改善するか」という問いへの答えは、Stage Dまで含めてyesである。

ただし改善は5 folds中2 foldsに限られ、400/773 wellsが悪化、worst wellは`+5.862833 ft`だった。
事前固定した安定性guardはfold数とworst-wellの2条件でFAILした。このFAILは「全体RMSEが悪化した」
という意味ではなく、「平均では改善したが、fold/well間で十分安定していない」という意味である。
このためtrain-side promotion、inference、submissionは行わない。

## Stage D実行設定

- Kaggle kernel: `kentookumura/exp286-geop-hmm-sparse-addonly-exp264-tvt-train`
- version / id_no: `1 / 127886849`
- Kaggle status: `COMPLETE`
- summary生成まで: `9,600.936秒`
- feature: clean base 273 + full13 compact 77 = 350列
- model: 1 variant × 3 LightGBM configs × 5 folds = 15 T4 boosters
- parent/control再学習: 0
- HMM/PF再生成: 0 / 0 well-runs
- inference / submission: 0 / 0

## parent12とのStage D比較

| 指標 | parent12 | full13 | delta |
| --- | ---: | ---: | ---: |
| pooled OOF RMSE | 8.460811 | 8.403784 | -0.057027 |
| near 0-250 | 1.583151 | 1.552892 | -0.030259 |
| mid 250-1000 | 4.099686 | 4.052844 | -0.046842 |
| 1000+ | 9.302283 | 9.241418 | -0.060865 |
| hidden-like spatial | 9.420315 | 9.249124 | -0.171191 |
| hidden-like typewell-purged | 9.341391 | 9.177678 | -0.163713 |

fold delta `full13 - parent12`は`+0.019882 / +0.141198 / -0.456150 / -0.123783 /
+0.105954 ft`で、fold 2と3の2/5だけが改善した。

773 wells中373 wellsが改善、400 wellsが悪化し、well delta中央値は`+0.025209 ft`だった。
worst well `2d35f86d`はparent12 `9.612724`からfull13 `15.475557`へ悪化し、
deltaは`+5.862833 ft`だった。

## guard判定

| 条件 | 判定 |
| --- | --- |
| pooled RMSE改善 | PASS |
| 3/5 folds以上改善 | FAIL（2/5） |
| near非悪化 | PASS |
| 1000+非悪化 | PASS |
| hidden-like非悪化 | PASS |
| worst-well回帰 `<= +0.25 ft` | FAIL（`+5.862833 ft`） |
| 総合 | FAIL |

比較対象は元のparent12 compact add-only `8.460811`である。fixed fallback `8.238332`はStage Bの
hard-selector診断値であり、Stage Dの350列LightGBMとの直接比較条件ではない。

## feature importance

15 modelsのgain合計ではcompact 77列が全gainの`76.4885%`を占めた。上位は両objective・両bankの
`top1_minus_anchor`と`beam_mean` scoreだった。名前に`geop_hmm`を含む3 compact特徴の合計は
全gainの`0.3996%`で、追加候補の直接特徴も未使用ではないが、改善の大半はbank全体のrank/margin/
aggregate表現を介している。

## Stage B/C履歴

- Stage B: hard selector `8.5870043867 -> 8.4777396073`、delta `-0.1092647794 ft`、3/5 folds。
  `geop_hmm`のID、availability、native confidenceを他候補と同様に追加し、score guardをPASSした。
- Stage C: nested hard selector `8.6525319556 -> 8.4486821528`、delta `-0.2038498028 ft`、4/5 folds。
  40 CPU models、25 partitions、77 compact features、score/leakage guardをPASSした。
- fixed fallback `8.2383315465`はStage B hard selectorより良かったため、hard pathの直接推論は不採用。

## 再現性と成果物

- Stage D metrics SHA: `1c4dfcd...4d2421`
- model manifest SHA: `1ad06b7e...0a22e`（15 model entries）
- OOF prediction SHA: `0769e600...ec0ae`
- reproducibility manifest SHA: `d77a6fb6...e774d`
- Stage C input manifest SHA: `cddfe5c6...3c7b5`
- output: `experiments/exp286_geop_hmm_sparse_addonly_candidate_on_exp264/kaggle/output/stage_d_v1`

大きなOOF/model本体はKaggle outputに保持し、ローカルにはmetrics、fold/bucket/hidden-like/by-well、
feature importance、model/reproducibility manifestだけを取得してSHAを照合した。

## 次

exp286の候補追加実験はStage Dまで完了した。平均改善は記録するが、安定性guard FAILのため
inference/submissionへは進めない。保存済みparent12/full13 OOFを用いたtarget-free tail-risk attributionを
別承認の0-booster readout候補としてバックログへ残す。
