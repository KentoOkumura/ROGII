# ROGII discussion insights

取得日: 2026-05-28

対象:

- Kaggle competition discussion の `top` / `recent` / `active` 上位トピック。
- 取得は Kaggle CLI API の `competition_list_topics` と `competition_list_topic_messages`。
- 個別全文アーカイブではなく、実験設計に使うための要約。

## 重要結論

1. この問題は row-level tabular regression ではなく、well ごとの prefix-conditioned geosteering / sequence tracking として扱うべき。
2. `TVT` は単なる深度や層厚ではなく、typewell に対応する相対的な地質位置の座標として考えるのが実装しやすい。
3. `GR` の point-wise matching だけではノイズが強い。NCC / DTW / DP / beam / PF は local matcher として使い、neighbor well 由来の structural guide と組み合わせるのが本筋。
4. `TVT_input` 既知区間、特に prediction start 直前の lateral GR は強い anchor。typewell GR より lateral prefix の自己相関が効く場合がある。
5. formation surface は強いが、train-only columns をそのまま推論特徴に使ってはいけない。fold-safe に spatial impute / plane fit して使う。
6. CV は well-level GroupKFold が基本。Public LB と CV に gap があっても、改善方向が一貫するかを見る。randomness と leakage に注意。

## データ・定義

- Host は competition data の PowerPoint を読むことを強く推奨している。水平井の軌道、GR、typewell を地質解釈として理解するのが前提。
- visible test は推論コード確認用の train 由来サンプルで、hidden test は提出時に差し替わる。train と hidden は基本 overlap しないが、typewell は共有され得るという認識。
- `TVT` は typewell 上の地質位置に対応する coordinate と見る。`XYZ` は lateral well の真の空間位置、typewell は vertical な地質定義。
- `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA` は formation top。議論では Z/TVD 空間の値として扱うべきという指摘があり、TVT 空間に直接混ぜると大きくずれる。
- Duplicate typewell は意図された挙動。typewell は近傍井や過去に掘った lateral interpretation から作られる pseudo-typewell の場合がある。
- 2026-06-07 audit note: train `__typewell.csv` の byte hash / 正規化 hash では exact duplicate は 34/773 wells に限られ、Discussion 697431/700827 周辺の「57/69 unique typewells」は exact duplicate としては再現しない。現時点では shifted GR similarity、PNG visual typewell ID、または supertype cluster 仮説として扱い、exact duplicate leakage と混同しない。

## GR matching

- Host の tip: prediction start 前の lateral GR は typewell GR より高解像度なことがあり、TVT が負方向に進む well では lateral prefix との correlation がより良い場合がある。
- Normal DTW は monotonic sequence 前提なので、reverse direction や fold-back を扱えないことがある。DWT/DTW を使うなら方向・segmentation・open boundary を明示的に考える。
- Dynamic Programming / Viterbi tracking は local smoothness constraint と GR matching をきれいに実装できるが、ある議論では OOF は改善しても LB はほぼ動かなかった。原因は DP ではなく point-wise GR observation cost が弱いこと、と整理されている。
- 使い方としては、NCC / DTW / DP / PF / beam を standalone predictor にするより、candidate path、confidence、residual、estimator divergence として tree model に渡す方が現実的。

## Spatial / geology

- Neighbor wells の dip は似る。formation dip / structural surface を地域的に滑らかな prior として使うべき。
- 上位議論の共通パターンは、structural guide + local matcher。formation plane / KNN / spatial imputer で大局的な TVT surface を作り、GR matching で局所補正する。
- Geophysicist の投稿では、global な TVT-Z correlation は強いが、within-well の lateral では TVT-Z slope がほぼゼロに近いと報告されている。`Z` だけを過信しない。
- Signed azimuth と `dZ/dMD` の組み合わせが重要。反対方向に掘る lateral は同じ線上でも地層を逆順に見る。
- Q-3D tortuosity は有望な domain feature。一方で Catch22 / ClaSP のような well-level generic time-series feature は GroupKFold で悪化したという negative result がある。

## Modeling

- Pure tabular GBDT は一定のところで壁に当たる、という議論が多い。ただし input formulation が良ければ single model でも 10 未満の報告がある。
- 有望な formulation:
  - `TVT_input` 末尾 anchor からの residual / drift を予測する。
  - prefix slope, recent GR, self-NCC, typewell NCC, spatial surface residual を入れる。
  - beam / PF / DP / deterministic DTW を candidate path feature として足す。
  - segmentation または multi-path prediction として、複数の plausible TVT trajectory を出す。
  - CNN / SegFormer / MDN / heatmap matching は研究価値あり。ただし synthetic pretraining が real well に transfer しない報告もあり、まず small CV で反証する。
- Test-time learning / online adaptation は参加者間で議論され、0.15-0.2 ft 程度の CV 改善報告がある。ただし取得範囲では host の明確な許可回答を確認できなかったため、使う場合は rules risk として扱う。

## CV / LB / submission

- well-level GroupKFold は必須。row split は leakage。
- Public LB と CV の gap はあり得る。議論では plain GBDT で CV around 11 / LB around 9.6 のような例があり、重要なのは gap の絶対値より改善方向の一貫性。
- Public 26% split は固定と見るのが妥当、という参加者見解。大きな score fluctuation は GPU、Numba、multiprocessing、未固定 seed、OOM などを疑う。
- Scoring error や異常に大きい score は、初期 scorer issue、submission format、NaN/inf、OOM、baseline fallback が原因候補。
- 提出前チェックは `id,tvt`、sample order、row count、duplicate id、NaN/inf、hidden rerun のデータパス、メモリを必ず見る。

## 次に試す実験

1. `exp002_drift_self_ncc`
   - `last_known_TVT` anchor からの drift target。
   - prefix slope、recent GR、lateral self-NCC、typewell NCC の最小構成。
   - GroupKFold by well、evaluation zone only。

2. `exp003_spatial_surface_guide`
   - formation top / ANCC を fold-safe に spatial impute。
   - signed azimuth、`dZ/dMD`、Q-3D tortuosity、neighbor well distance を追加。
   - structural surface prediction と model prediction の residual を見る。

3. `exp004_tracker_features`
   - DP / Viterbi、beam、PF、deterministic DTW を feature-only で追加。
   - OOF 改善だけでなく fold variance と LB を確認し、local matcher 過学習を検出する。

4. `exp005_segmentation_or_multipath`
   - direction segment、top-k path、candidate trajectory confidence を小さく試す。
   - deep model はまず feature extractor / auxiliary path generator として使う。

5. `exp006_submit_robustness`
   - deterministic seed、single-thread / Numba seed、OOM fallback、submission validation を固める。

## 参照トピック

- Welcome / hidden test note: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697416
- Dataset issue fixed: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697400
- Diagram of the problem: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697418
- TVT definition: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698282
- Geological formations: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697406
- Geologist tips: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698825
- Duplicate typewells: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698449
- DWT / time warping: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697431
- UI visualizer / segmentation discussion: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/700424
- Multi-trajectory CNN: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853
- Paradigm shift from tabular: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699289
- DP / Viterbi tracking: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702919
- Geophysicist domain priors: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702131
- Surface columns in TVD: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701034
- Online learning / test-time tuning: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/698002
- CV/LB correlation: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701691
- Public LB fixedness / randomness: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/701995
- Submission scoring: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/697329
- Submission format issues: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/702308
