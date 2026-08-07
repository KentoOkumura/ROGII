# exp390_parallel_strip_surface_registration_readout

## 状態

- ルート: `pf_beam`
- 状態: Kaggle CPU Stage 0 preflight完了・sparse two-sided supportでFAIL close
- CV / Public LB / Private LB: 未実行 / 未提出 / 未提出
- 作成日: 2026-07-24
- 親・control: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 実行順: 当初はexp383結果確認後。2026-07-24のユーザー実行指示で固定値を変えず
  16-well Stage 0だけ先行
- 実装・正規train Notebook採用: 承認・完了
- package / push / 16-well Stage 0 preflight: 完了
- 773-well full run / Stage 1 / 2 / inference / submission: Stage 0 FAILでblocked

## 仮説

XY上で近接するhorizontal wellは、ほとんどが少し横へ平行移動した長い直線軌跡である。
この構造をwell-levelの近傍距離やazimuth featureとして使うだけでなく、queryごとの
along-track `s` / cross-track `n`座標へ登録し、同じ`s`にあるouter-train donorの
`S=TVT+Z`を`n`方向へtwo-sided補間すれば、exp226のgeneric XY donor fieldより
安定した物理TVT pathを作れる。

## 変更点

- query XYのPCA軸をcanonicalなalong-track方向とし、近傍wellを共通`s`へ対応付ける。
- `angle<=5°`、overlap`>=0.80`、cross-track`<=2000 ft`を満たすouter-train donorだけを使う。
- 同じ`s`のdonor `S=TVT+Z`へ`S=a(s)+b(s)n`のHuber local-linear fitを行う。
- queryの既知prefixからvertical gauge interceptを1個だけfitする。
- 正負両側、4 donor wells以上の行だけstrip predictionを使用し、その他は保存済みexp226 OOFへexact fallbackする。
- target GR、生Formation、suffix truth、nearest residual/GR copy、soft blend、selectorは使わない。

## 検証方針

- Fold: exp226と同じstable outer 5-fold
- Group: `well_id`
- Score rows: outer-valid unknown suffix
- Stage 0: target-free role-read、pair/support coverage、16-well runtime/RSS
- Stage 1: known prefix末尾512行のrolling-origin reconstruction
- Stage 2: prediction/SHA freeze後だけsuffix truthをjoinし、保存済みexp226 OOFと比較
- Leakage Check:
  - outer-valid wellはdonor/pair/fitから完全除外
  - freeze前のtarget suffix truth / raw Formation / GR readは0
  - visible 3 sample wellsによる設定選択なし

## 実行入口

- 学習 notebook: `exp390_parallel_strip_surface_registration_readout_train.ipynb`
- 推論 notebook: `exp390_parallel_strip_surface_registration_readout_inference.ipynb`
- compact train候補:
  `exp390_parallel_strip_surface_registration_readout_compact_selfcontained_train.py/.ipynb`
- compact inference候補:
  `exp390_parallel_strip_surface_registration_readout_compact_selfcontained_inference.py/.ipynb`
- 正規trainはcompact候補を採用済み。正規inferenceはtemplate scaffoldのまま。
- private CPU / internet offの16-well Stage 0 preflightまで承認済み。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | 未実行 |
| Public LB | 未提出 |
| Private LB | 未提出 |

version 1はKaggle input resolverが3件の`test/`を選択し、本体計算前に停止した。
resolver修正後のversion 2は`COMPLETE`したが、two-sided row / well coverageと
donor p05が`0 / 0 / 0`でStage 0をFAILした。16 wells中eligible pairを持つqueryは
8本、queryあたり最大2 donorで、必要な4 donorかつ正負両側supportを満たすnodeは
0だった。leakage/read、fallback、angle、overlap、runtime、RSS gateはPASSした。

## 所見

### 良かった点

- 設計段階で、平行性をsimilarity featureではなくsurface補間の座標系として定義した。
- exp114/119/201のnegative evidenceを踏まえ、残差・GR・nearest TVTの直接コピーを避けた。
- exp383とはno-formation / explicit same-s registrationとして科学差分を固定した。

### 悪かった点

- 16-well preflightではtwo-sided supportが完全に退化し、valid strip fitは0だった。
- wellboreが平行でも地層が連続・平行とは限らない。

### リスク / 注意

- fault、landing zone差、edge-of-familyではcross-track補間が破綻し得る。
- train spatial densityとhidden test densityの差でCV/LBが乖離し得る。
- sparse supportだけで事前gateをFAILしたため、surface accuracy評価前にbranchを閉じた。
- FAIL後のthreshold/bandwidth/one-sided/soft-blend救済は禁止する。

## 次

exp390はStage 0 FAILで終了し、773-well full run、Stage 1 / 2、inference、
submissionへ進まない。再検討する場合も、exp390のthreshold救済ではなく、
別の0-fit target-free support censusから始める。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて日本語優先で記録する。
