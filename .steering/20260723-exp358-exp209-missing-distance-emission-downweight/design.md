# 設計

## アプローチ

trusted exp209 exact-HMMの観測面をそのまま使い、raw GRが欠損して補間値となったrowだけ、
最寄りraw finite GRまでのrow距離でevidenceを弱める。exp307のfinite-MAD sigmaを
持ち込まないため、sigma縮小とmissing downweightの同時変更を避ける。

旧exp308 sourceは履歴参照に留め、新expの実装時はexp209 compact科学契約を基準に
self-contained化する。Stage 0はweight surfaceのtechnical/non-degeneracy監査だけで、
精度主張はしない。Stage 1は別承認された1 variantだけを実行する。

2026-07-25の実装ではStage 0だけをself-contained化する。missing mask / distance /
weightはwhole-well raw GR上で計算してunknown suffixへsliceし、補間GRはexp209と同じ
両方向linear interpolation + all-missing時Type Well GR平均fallbackとする。
unknown-suffix truth、fold、hidden-like、saved exp209 predictionはStage 0では読まない。

Stage 1追加承認後は、exp209直結の保存control、reporting fold、hidden-like assignmentを
SHA固定で読む。exact-HMM骨格はexp389の検証済みexp209 absolute-TVT実装を参照し、
Huber emissionは持ち込まず、exp209 capped Gaussian
`-0.5 * min(z^2, 600)`へStage 0でfreezeしたmissing-distance weightを1回だけ掛ける。
旧exp308は式と履歴の参照に限定し、exp307 finite-MAD sigma/control依存は持ち込まない。

## 実験範囲

- 対象実験: `exp358_exp209_missing_distance_emission_downweight`
- Route: `pf_beam`
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 変更する変数: missing row Gaussian log-emission multiplierだけ
- 固定する変数: exp209 interpolation、sigma、typewell、step 0.35、41 rates、
  `sig_r=0.002`、`sig_p=0.02`、momentum 0.998、prior、posterior mean
- weight:
  - raw finite observed row: 1.0
  - missing row: `max(0.25, 2^(-d/8))`
  - raw finite GRがwell内にない場合: 0.25
- Stage 0: technical audit 1 / HMM・model・booster 0
- Stage 1予約: 1 variant / 773 HMM runs / saved exp209 control / control rerun 0
- reporting: 5 folds、raw observed/missing、missing fraction、MD distance、
  hidden-like 2面、by-well、fixed LikPF/HMM 50:50

## 再現性設計

- seed policy: RNGなし、raw row/well順を固定
- stochastic処理: なし
- PF/Beam/likelihood-PF: control readout以外の再生成なし
- CPU/GPU: Kaggle CPU、internet/GPU/TPU off
- input SHA: Stage 0はraw well identity、Stage 1時にexp209 exact HMM cache/controlと
  fold/hidden-like assignmentsを追加
- feature SHA: missing mask、distance、weight、interpolated GR、Stage 0 summary
- prediction SHA: Stage 1時にraw/decompressed/logical content SHAを保存
- model SHA: fitted modelなし。decoder contract SHAを保存
- submission SHA: 非該当
- bootstrap: package時にconfig/sourceのloose/package/bootstrap一致を確認
- exact-HMMはRNGなし。well、row、grid、rate、variant順を固定し、
  predictionとweight contractのraw/decompressed/logical SHAを記録する。

## リスク

- 科学リスク: exp339はgap uncertaintyのplacement transferを2/5 foldsしか通さなかった。
- 観測リスク: exp346はfinite observed scale導入でdirectを1.356739 ft悪化させた。
- tailリスク: missing evidenceを弱めるとgeometry priorのwrong modeを修正できなくなる可能性。
- leakage: Stage 0はmissing surface、Stage 1はsurfaceとpredictionをsuffix
  truth/error結合前にfreezeする。
- runtime: Stage 1は773 HMM runsで約数時間を見込む。

## 実行後の結論

Stage 1は`17475.557881 sec`で完了したが、overall `-0.074283 ft`、
0/5 folds、1000+ `-0.082776 ft`、hidden-like 2面
`-0.224970 / -0.229587 ft`、by-well p95 `+0.469370 ft`、
worst `+6.630365 ft`、fixed blend `+0.036981 ft`で固定gateをFAILした。

距離減衰はlong-gapだけでなく全gap bucketを悪化させ、HMM smoothingを通じて
observed row側も悪化させた。half-life/floor gridやhard maskへ広げる根拠はなく、
設計どおりrescueなしで閉じる。
