# 設計

## アプローチ

初回は augmentation model を学習せず、raw train well を入力に変換 engine の物理・再現性
contract を固定する。厳密変換は inverse consistency を主評価とし、近似変換は
official-start anchor で変位を 0 にして known prefix を保持する。近似 TVT path から同じ
typewell の GR を線形補間し、raw 座標だけを変えた不整合 view を禁止する。

real well ごとの MD step、XY/Z slope、XY/Z curvature、TVT slope、typewell coverage を集計し、
real 分布の固定 quantile envelope から外れる synthetic view を reject する。TVT slope は
augmentation 採否の診断にのみ使い、model feature にはしない。後続 train を行う場合は
outer-train well だけから envelope を fit し、outer-valid は clean official-start view のまま
評価する。

初回の変換は transform kind ごとに 1 view / well とし、parameter は
`sha256(seed, well, transform_kind, view_slot)` から固定候補を選ぶ。全 transform の直積や
複数 warp の合成は行わない。

## 後続学習stage

exp251のcorrected 295列版は別runで学習中のため、その完了生成物を固定clean controlとして参照し、
exp259内では再学習しない。学習入力はcompleted version 3 feature auditのselected schema SHAへ固定し、
clean controlのmetricsはfit後の比較にだけ使うため両kernelを並列実行できる。outer-fold、base-row
sampling、LightGBM random seedはexp251 controlと同じnamespaceを使い、変更点をouter-trainへの
exact datum view追加だけに限定する。

selected 295列schemaを監査した結果、raw X/Yとlocal geometry列は0列だった。したがってrigid XY
translation/reflection/yawはcandidate-long featureを変えず、追加するとsample weightだけが変わる。
これら3変換はstrict inverse監査に残すが学習行にはしない。学習対象はTVT datum shiftだけとする。

全773 wellsをstable SHAで順位付けし、正確に25%をsynthetic view対象にして、well単位で
`-40/-20/+20/+40 ft`を1つ選ぶ。各foldではそのうちouter-trainに属するwellだけを追加し、
clean training rowsは全て保持する。raw-wellでのdatum equivarianceは
初回監査済みなので、学習時はselected candidate-long surfaceへの誘導作用を適用する。具体的には
last-known、candidate mean、3 absolute candidate source、candidate TVT、view candidate meanの7列だけを
同じ量だけshiftする。残り288列、candidate error、within10 labelは完全一致をhard guardする。
outer-validはclean viewだけをscoreする。

`md_stretch`を含む近似5変換はこの学習stageでは無効とする。PF/HMM/geometry candidateとdiagnosticを
近似pathに合わせて再生成する別compute contractなしに、raw pathだけを変えたtraining rowは作らない。

## 実験範囲

- 対象実験: `exp259_coordinate_equivariance_path_warp_augmentation`
- Route: `ensemble`
- 親実験: `exp251_raw_test_safe_dual_objective_candidate_ranker`
- 変更する変数: raw well への coordinate equivariance / path warp view と整合的な
  GR・geometry・prefix・spectral diagnostic 再生成。
- 固定する変数: raw train files、official-start 定義、typewell reference、変換候補 grid、
  seed 42、validation well、既存 candidate/model prediction。
- 初回監査実行量: 0 variant / 0 model config / 0 fold / 0 booster。
- optional学習実行量: 1 variant / 2 objectives / 5 folds / 10 CPU boosters。PF/Beam/HMM、
  exp218、exp251 parent/controlは再学習しない。

## 再現性設計

- seed policy: `sha256(seed, well, transform_kind, view_slot)` の先頭64bitから候補 index
  を決める。Python `hash()` と global RNG は使わない。
- stochastic 処理の有無: parameter choice と spline control-point sign のみ擬似乱数的だが、
  stable key から完全に決定する。
- PF/Beam / likelihood-PF / seed bagging の有無: 初回はすべて新規実行なし。
- 並列処理と乱数の関係: well / transform を sort し、各 view の parameter は独立 key で
  決まるため thread scheduling に依存しない。初回 notebook は単一 process。
- CPU/GPU runtime と deterministic flags: Kaggle CPU、GPU false、internet false。
- train cache / test feature regeneration の SHA 記録方針: raw input file SHA、transform
  manifest decompressed SHA、envelope / summary SHA、row/well/transform countを保存する。
- model manifest / prediction / submission SHA 記録方針: 初回は model / prediction /
  submission を生成しない。後続 train stage を追加する場合に別途記録する。
- Kaggle package bootstrap 確認方針: prepare 後に embedded config、`src` helper、CPU /
  internet metadata、canonical slug を確認してから push する。

## リスク

- リークリスク: validation true TVT/error/oracle で transform parameter や採否を選ぶと
  oracle augmentation になる。初回は分布 contract 監査だけとし、後続では outer-train
  envelope だけを使う。
- CV/LB 不一致リスク: synthetic geometry が train 分布内でも hidden test の path 分布と
  一致する保証はない。clean official-start OOF と hidden-like / 1000+ / worst-well guard
  通過前は inference / submit しない。
- ランタイム/メモリリスク: 全773 wells × 9 transformのfull CSV保存は大きい。viewは1件ずつ
  処理し、well-level manifestと小さいpreviewだけ保存する。
- 再現性リスク: scipy splineやglobal RNGに依存するとversion/順序差が出る。control-point
  warpは固定piecewise-cubicではなくNumPy線形補間＋固定smooth kernelで実装し、stable keyと
  content SHAを保存する。
