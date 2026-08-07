# exp259_coordinate_equivariance_path_warp_augmentation 結果

## 状態

Kaggle CPU transform audit version 1と、`md_stretch`を除外したfull-well exact datum
学習version 1はともに`COMPLETE`です。学習自体とequivariance contractは正常に完了しましたが、
exp251 clean controlとの事後比較でhidden-like 2面と最大well回帰guardを通過しませんでした。
最終判定は`completed_train_side_guard_failed_no_inference_no_submit`です。

- audit kernel: `kentookumura/exp259-coordinate-equivariance-warp-audit-train`
  （ID `127328846`、version 1）
- train kernel: `kentookumura/exp259-exact-datum-fullwell-train`
  （ID `127436131`、version 1）
- train runtime: `4776.873 sec`（約79.6分）
- active variant / model config / fold / booster: `1 / 2 / 5 / 10`
- rows / wells / candidates / selected features: `3,783,989 / 773 / 11 / 295`
- synthetic wells: `193 / 773`（24.97%）
- Public LB / Private LB: なし / なし

## 仮説と親実験との差分

親はexp251のraw-test-safe 295列dual-objective rankerです。candidate bank、features、fold、
sampling seed、LightGBM設定は固定し、outer-trainへのexact TVT datum view追加だけを変更しました。
absolute datum表現への過適合を抑えてclean OOF、とくにlong-tailを改善する仮説を検証しました。

## clean control比較

比較対象は別kernelで完了したexp251 version 4
`raw_test_regenerated_copcf`の保存済み295列clean OOFです。exp259内でcontrolは再学習していません。

| 指標 | exp251 clean | exp259 exact datum | delta | guard |
|---|---:|---:|---:|---|
| fixed-Viterbi RMSE | 8.502212005 | 8.427125551 | -0.075086454 | PASS |
| fixed-Viterbi MAE | 5.010195640 | 4.982795759 | -0.027399882 | 参考 |
| fixed-Viterbi within 10 ft | 0.849685345 | 0.849401254 | -0.000284092 | 参考 |
| path switches | 7,107 | 7,053 | -54 | 参考 |
| candidate logloss | 0.327735530 | 0.327708967 | -0.000026563 | PASS |
| distance 1000+ RMSE | 9.326545505 | 9.244368080 | -0.082177424 | PASS |
| exp115 spatial RMSE | 8.788133228 | 8.855452660 | +0.067319432 | FAIL |
| exp115 typewell-purged RMSE | 8.746112958 | 8.791490214 | +0.045377256 | FAIL |
| 最大well回帰 | - | `aed44918` +6.370552990 | +6.370552990 | FAIL（上限+0.25） |

fixed-Viterbiのoverall、candidate logloss、1000+は改善しました。一方で、hidden-like
spatialとtypewell-purgedがともに悪化し、最大well回帰は許容値の約25倍でした。6 guard中
3 PASS / 3 FAILのため、事前契約どおり採用しません。

なお1000+のPASSはexp251 saved controlに対する相対改善です。exp251で固定した絶対上限
9.234366423に対してはexp259の9.244368080が+0.010001658上回っており、絶対guardは未達です。

### fold別比較

| fold | exp251 clean | exp259 exact datum | delta |
|---:|---:|---:|---:|
| 0 | 7.653335511 | 7.719531192 | +0.066195681 |
| 1 | 9.472239430 | 9.384259812 | -0.087979619 |
| 2 | 8.075131903 | 7.947896045 | -0.127235858 |
| 3 | 9.017009135 | 8.814951528 | -0.202057608 |
| 4 | 8.163005322 | 8.158448432 | -0.004556890 |

4/5 foldsで改善しましたがfold 0は悪化しており、well別386改善 / 384悪化と合わせても
安定した一様gainではありません。

## 学習結果

| mode | RMSE | MAE | within 10 ft | oracle label accuracy | path switches |
|---|---:|---:|---:|---:|---:|
| probability rowwise | 8.556052667 | 5.164365596 | 0.852288947 | 0.168975121 | 1,003,205 |
| expected-error rowwise | 8.532211478 | 5.041134407 | 0.847643849 | 0.189872381 | 842,877 |
| expected-error fixed Viterbi | **8.427125551** | **4.982795759** | 0.849401254 | 0.199173941 | 7,053 |

candidate AUCは`0.924427649`でcontrol比`-0.000031088`、expected-error MAEは
`4.586597201`で`+0.032752211`悪化しました。candidate logloss/Brierはごく小さく改善した
一方、candidate errorの絶対値校正は悪化しています。したがってoverall gainは候補単体の
一様な改善ではなく、固定Viterbiが選ぶpathとswitch構造の変化による寄与が大きいと解釈します。

## well別の安定性

- 改善386 wells、悪化384 wells、同値3 wellsで、改善は広く一貫していません。
- 最大改善は`389ae58f`の`-6.229024405`。
- 最大回帰は`aed44918`の`7.248608 -> 13.619161`、`+6.370552990`。
- 絶対RMSE最大は両variantとも`fb03ae90`で、exp259は`58.021328890`、
  controlは`58.004236030`です。これは「最大絶対RMSE」と「最大の差分回帰」が別wellで
  あることを示します。

## exact datum実装・再現性guard

全5 foldsでcandidate error、within10 label、相対特徴288列の不変性がPASSしました。
absolute TVT特徴7列の最大shift誤差は`0.0`で、tolerance `1e-5`以内です。fold別の
augmented wellsは`158 / 147 / 156 / 152 / 159`、augmented long rowsは
`166,628 / 155,441 / 160,820 / 158,202 / 164,846`でした。
`md_stretch`を含む近似5変換と、feature-identicalなrigid XY 3変換は学習していません。

学習ログ中のLightGBM feature-name warningは学習停止やschema不一致を示すものではありません。
295列schema SHA、10 model manifest、全foldのequivariance guardは一致しています。

## transform auditの履歴

773 wells × 9 transforms = 6,957 viewsを監査し、6,129 viewsを採択しました。厳密4変換は
全件通過し、inverse最大絶対誤差`9.313225746154785e-10`、local metric相対差最大`0.0`です。
`md_stretch`はMDだけを0.97/1.03倍してXYZを据え置く定義がtrajectoryのMD微分量と
整合せず、773/773 viewsがreal-train geometry envelope外となったため除外しました。

## 生成物とSHA

小さいmetrics、by-well、feature schema、model manifest等だけを
`kaggle/output/train_variant0_v1/artifacts/`へ取得しました。取得済みartifactはsummary記載SHAと
一致しました。

- training summary SHA256:
  `74909725f644daea693320f61972d9c5b4ac85b6440c44bec4ab39194a2995e7`
- selected feature schema SHA256:
  `7a9217d6ed96f5f1e569dbefff2a1fb17751405d6ddccae5e5d9dbf12da787ae`
- model manifest SHA256:
  `7f6cd5c84dd7693c271359537ace1e359af2ecffd39cf6e7fa5e446c7d326d4b`
- OOF decompressed content SHA256（Kaggle log/summary一致）:
  `96d34a2f7eb68f576f6a7e51bec8c11b6ef7294ec8989442d9fda8d644290913`

監査時にmodel本体、imputer、OOFを`/tmp`へ選択取得し、診断生成物15/15、model 10/10、
imputer 5/5、OOF decompressed content SHAを独立再計算して全一致しました。不採用runのため
これらの大きな本体は実験配下へ保存せず、小さい診断生成物だけを保持しています。
inference predictionとsubmissionは生成していません。

## 判定

exact datum augmentationはoverallとlong-tailに有効な信号を示しましたが、hidden-likeと
well-level安定性を犠牲にしています。事前guardを緩めずtrain-side rejectedとし、
inference・competition submissionへ進みません。比率やshift幅の事後gridで救済する実験も
直ちには行いません。再訪する場合は、今回改善した1000+だけを対象にsynthetic比率を下げた
独立variantとして、同じhidden-like／最大well回帰guardを維持して検証します。

## 次のアクション

exp259のinferenceとsubmissionは行いません。改善386 wellsと悪化384 wellsのtarget-free属性差を
先にreadoutし、long-tail限定ruleをouter-trainだけで固定できる証拠が得られた場合に限り、
`longtail_only_exact_datum_augmentation`を別variantとしてユーザー確認後に検討します。
