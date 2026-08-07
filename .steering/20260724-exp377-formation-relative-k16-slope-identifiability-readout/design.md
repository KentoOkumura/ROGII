# 設計

## アプローチ

構造座標を `S = TVT + Z` と置く。各地層面 `F_f` についてouter-train donorの16区間ごとに `d(S-F_f)/dMD` を作り、exp226と同じXY近傍kernelで対象坑井位置へ補間する。別途outer-trainだけから `FormationPlaneKNN(k=10)` で対象位置の地層面を推定し、その `dF_f/dMD` を足し戻して `dS/dMD` を復元する。

6地層系列を個別報告し、primaryは事前固定した6系列のrobust medianとする。この段階ではTVT予測候補、HMM、PF、ML特徴を作らず、rateと累積pathの識別可能性だけを測る。

2026-07-24の実装では、区間rateをfiniteかつ正の`ΔMD` step rateのmedianに固定する。exp226と同じ方位`118.4°`、`|projection|>0.3`、XY最近傍50、bandwidth 500 ft、ridge 1を使い、donor rateをprojectionで割って補間後にtarget projectionを戻す。地層面はouter-train坑井ごとのXY・地層面medianからk=10局所平面をfitし、target K16区間の両端差から`dF_f/dMD`を得る。局所平面が数値的に解けない場合だけ同じk=10の逆距離平均へfallbackする。

Stage 0はvalid roleの`TVT`・6地層列read count 0、source/valid overlap 0、全予測とSHAをtruth前にfreezeする。Stage 0 PASS後だけraw `TVT`を別readerでlate joinし、segment rateとprefix末尾`S`をanchorに積分したpathを評価する。

## 実験範囲

- 対象実験: `exp377_formation_relative_k16_slope_identifiability_readout`
- Route: `pf_beam`
- 親実験: `exp226_tvt_slope_kriging_hmm`
- 変更する変数: 補間対象をTVT/K16直接勾配から6種類のformation-relative勾配へ変更する。
- 固定する変数: exp226のouter 5-fold、K=16、XY donor kernel、対象坑井除外、評価scope。
- Primary: `robust_median_across_6_formations`
- 出力: 6地層別rate/path、median rate/path、support/fallback診断。
- 実行量: 1 diagnostic / 6 reporting surfaces / 5 folds / model・HMM・PF・booster各0。
- 実装境界: 別名compact self-contained train/inference候補までを初回承認とし、追加の実行指示で正規train Notebook採用とKaggle CPU v1だけを実施する。

## 段階と停止条件

1. Stage 0で行数・坑井数・区間数・対象側read count・coverage・supportを監査する。
2. Stage 1でexp226の直接rate/pathと比較する。
3. 受け入れ基準を1つでも満たさなければexp378、exp379、exp380を停止する。
4. 有利な地層のpost-hoc選択や閾値救済は行わない。

Kaggle CPU v1ではStage 0の`effective_donors_p05=2.59469484575288`が固定下限10を
満たさず、truth join前に停止した。ほかのStage 0 checkはすべてPASSしたため、
設計どおりStage 1を開かずbranchを閉じる。

v1後のコード監査で、exp226型direct controlと6 formation-relative fieldは同じ
eligible donor XY inventory、近傍50、bandwidth 500 ftを共有し、effective-donor数が
treatment固有の識別条件ではないことを確認した。ユーザー選択によりv2では
`effective_donors_p05`の数値と閾値判定を保存したままreport-onlyとし、ほかの
integrity checkを全PASSした場合だけtruthをlate joinする。これはparameter救済ではなく、
共通kernel上のcontrolled comparisonを本来のStage 1まで開く設計修正である。

## 再現性設計

- seed policy: 乱数を使わない決定論的幾何計算。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: なし。
- 並列処理と乱数の関係: 乱数なし。並列順序に依存しないwell id順へ整列する。
- CPU/GPU runtime と deterministic flags: CPUのみ、GPU学習なし。
- train cache / test feature regeneration の SHA 記録方針: fold manifest、入力schema、地層面、6系列、median系列についてdecompressed content SHAを記録する。
- model manifest / prediction / submission SHA 記録方針: model・submissionなし。readout artifactとsummary SHAを記録する。
- Kaggle package bootstrap 確認方針: 実装時にoffline import smokeを行うが、この設計ターンではpackage化しない。

## リスク

- リークリスク: 対象坑井の生Formation列や正解TVTを参照すると即時leakになるため、role別read guardを必須にする。
- CV/LB不一致リスク: 地層面欠損やtest分布外座標で補間が崩れる可能性があるためfallback/support分布をfold別に記録する。
- ランタイム/メモリリスク: 6面×16区間の中間表を全行展開せず、well-segment表で計算してから必要なreadoutだけ展開する。
- 再現性リスク: donor同距離tieの順序をwell idで固定し、集約順を固定する。
