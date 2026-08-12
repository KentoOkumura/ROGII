# 設計

## アプローチ

exp295と同じunary`u_theta(i,k)`とfixed exp209 grammarを使うが、structured lossを256-row window内だけで計算する。window truthをsigma`0.35 ft`のGaussian label emission`ell_y`へ変換し、

`L_window = (log Z(u) - log Z(u + ell_y)) / N_window + 0.25 * L_local_CE`

を学習する。通常posteriorとlabel-conditioned posteriorのforward-backwardはwindow内だけで行う。評価時はwindowへ分割せず、official prefixからhidden suffix全体をfixed exact SSMで一度にdecodeする。

## Window contract

- length: 256 scored rows。suffixが短い場合だけ実長maskとし、paddingはlossへ入れない。
- count: 各outer-train well・各epochで3 scheduled slots、最大3 active windows。
- window 1: official hidden suffixの先頭から開始。
- window 2/3: official hidden suffix内のeligible startをstable SHA256(`fold, well, epoch, slot`)順で決め、同epoch内は重複・overlapさせない。eligible startがなければslotをinactive zero-lossとし、window 1を複製しない。start/active scheduleをtraining前にmanifestへfreezeする。
- selectionにはTVT値、error、formation、candidate、oracleを使わない。
- interior windowのencoder contextはofficial visible prefixだけ。window直前truthはencoder inputにせず、initial TVTのGaussian boundary emissionと直前2 rowから量子化したrate-state priorとしてloss初期化だけに使う。
- boundary supervisionはtraining-only。outer-valid/current-testはofficial prefix hard clampとexp209 initial-rate grammarだけを使う。

最大556 fit wellsなら1 epochは最大1,668 active windows / 427,008 scored positionsで、exp295 version 3の8,571,405 suffix positionsの約4.98%を上限とする。4 DP sweepsは残るため、実測microbenchmarkを省略しない。

## 固定する科学契約

- input/architecture/preprocessing/decoder/controlsはexp331設計およびexp295と同一。
- decoderは`step=0.35`、41 rate states、rate span 0.10、`sig_r=0.002`、`sig_p=0.02`、`start_sig=0.75`、`r0_sig=0.01`、`band_pad=100`、`mom=0.998`。
- optimizerはAdamW、lr`3e-4`、weight decay`1e-4`、最大8 epochs、gradient clip 1.0、AMP、1 well-window/batch、gradient accumulation 4、worker 0。
- stable outer-train holdoutの同じwindow objectiveでearly stoppingする。outer-valid scoreはepoch選択に使わない。
- local CEのみ、full-well structured training、windowサイズ比較は同じexpへ含めない。

## 計算量gate

実装承認後、suffix長quartileごと4件の固定16 windowsで、normal/label-conditioned forward-backward、backward、optimizer、full-well real/control decodeをT4実測する。manifest上の全window cellsとfold 0 evaluation cellsへ外挿し、peak`<=14 GB`、保守的fold総時間`<=8.5 h`を必須とする。exp295実測比だけの推定でpushしない。FAIL時はwindowを128へ縮めたり1--2個へ減らしたりせずcloseする。

## 検証設計と段階実行

Stage A/B/Cのfold、score rows、controls、technical/GR evidence/TVT/safety/promotion gateはexp331と同じにする。

### Stage A

- `1 architecture × fold 0 × seed 42 = 1 neural model`。
- LightGBM config/fold/booster、PF/Beam、parent/control再学習はすべて0。
- microbenchmark PASS後に別承認された場合だけfull foldを実行する。
- real exact-SSM RMSE/NLL/massとshuffle/geometry-only/saved exp209を比較する。window lossだけの改善をpromotion根拠にしない。

### Stage B

Stage A全PASSと別承認後だけfold 1--4を追加する。pooled OOF`<=6.0 ft`とGR attribution/hard-tail/well guardを全PASSした場合だけStage C候補とする。

### Stage C

Stage B PASSと別承認後だけ同じexp内でfull-well current-test exact SSMを実装する。5 fold modelsを独立decodeし、fold別posterior-mean TVTをrow-wise等重み算術平均する。unary平均とfold weightingは行わず、training window境界やteacher boundaryをcurrent-testへ持ち込まない。

## 実験範囲

- 対象: `exp332_prefix_gr_unary_fixed_window_structured_ssm`
- Route: `ensemble`
- 親: `exp295_prefix_anchored_wholewell_gr_alignment_ssm`
- 変更する変数: structured objectiveのsequence supportをcomplete-wellから固定256-row・3 windows/well/epochへ限定する。
- 固定する変数: objective family/weight/sigma、input、fold、architecture、preprocessing、decoder、controls、full-well evaluation gate。
- 優先順位: exp331を先行。exp331が進行中またはpromotionした場合は本案を実装しない。

## 2026-07-22 実装承認の反映

- exp331はStage Aで`real_rmse_vs_exp209`、well p95、worst-well gateをFAILし、Stage B・推論・提出をrescue gridなしでclose済み。このためexp332の先行条件は成立した。
- 今回の実装は`implementation_only`に固定する。Stage 0は固定16-window T4 microbenchmark 1件、一時benchmark neural model 1、Stage A永続model 0であり、別承認前はpackage/pushしない。
- Stage Aを将来承認する場合も`1 architecture × fold 0 × seed 42 = 1 neural model`、LightGBM config/booster、PF/Beam、parent/control再学習はすべて0のままとする。
- 正規Notebook scaffoldは上書きせず、compact self-contained Jupytext候補を別名で生成する。採用・Kaggle実行は別承認とする。
- window schedule manifestはtruthを読む前に全epoch分を固定する。teacher boundary manifestはschedule固定後にtraining truthから作り、encoderへ渡さずwindow StateSpecの初期priorにだけ使う。

## 再現性設計

- global seed 42 + stable SHA256 window schedule。manifest/content SHAを学習前にfreezeする。
- CUDA/AdamW/dropoutはstochastic。worker 0、CuDNN benchmark false、deterministic algorithms`warn_only=True`、AMP有効。
- deterministic anchorにはしない。
- input/fold/window/boundary manifest、feature schema/content、model、unary/window posterior/full posterior/prediction、Kaggle package/kernel versionのSHAを記録する。

## リスク

- teacher boundaryによりwindow内学習が容易になり、full-well accumulated driftを過小評価する可能性がある。
- window境界をまたぐ長距離整合性は学習されない。
- exact DPは残るため、20倍程度のrow削減でも8.5時間を超える可能性がある。
- exp331より実装・数値監査が複雑で、第一選択にはしない。

## 2026-07-22 Stage 0結果

- Kaggle T4 version 1、固定16 windows、temporary neural model 1、永続model/booster/PF/Beam/control再学習0で完走した。
- 保守的fold外挿`13.151137275 h`は固定上限`8.5 h`をFAIL、peak memory`1.203262806 GB`は`14 GB`上限をPASSした。
- decisionは事前登録どおり`close_without_window_or_loss_rescue`。Stage A/B/C、推論、提出、window/loss/decoder/architecture救済へ進まない。
