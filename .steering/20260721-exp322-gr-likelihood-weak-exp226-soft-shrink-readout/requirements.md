# 要件

## 2026-07-21 追加依頼

ユーザーからexp322の実装開始指示を得た。凍結済み設計を変更せず、train-side 0-model readout、compact self-contained Jupytext source、別名Notebook、unit test、静的検証までを実装する。Kaggle package/push/run、inference、submissionは引き続き対象外とする。

## 2026-07-21 実行追加依頼

ユーザーからKaggle CPUでの実行指示を得た。承認対象は`1 candidate / 1 matched control / 5 exp263 readout strata / 0 model / 0 booster / 0 parent rerun`。version 1は計算開始前にexp226元OOF foldとexp263 readout foldが一致しないことを検出して停止した。exp263を親として比較するためreadout strataは保存済みexp263 outer foldを正とし、exp226元foldは各wellで一意な5-fold OOF source identityとして別監査する。split、予測、threshold、gate、科学guardは変更しない。

## 依頼

PF/HMM 系の固定 blend で GR matching が曖昧な区間だけ、提出実績のある exp226 K16 予測へ保守的に寄せる案を、バックログ、steering、実験ディレクトリへ切り出して設計確定する。今回は設計と記録だけを行い、実装、Notebook 作り替え、Kaggle package/push/run、inference、submissionは行わない。

## 背景

- exp263 の固定 `exp226_w500_50_50` は `0.50*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm`、OOF RMSE `8.238331`、Public LB `7.800`で、PF/Beam route の再利用可能な提出済み基準である。
- exp280 は exp226 geometry 周囲の固定13 shiftをraw GR/typewell emissionで順位付けし、top1/top3/MRR/signがstable shuffledを5/5 foldsで上回った。一方、top1 `0.189547`、sign `0.498523`であり、hard shift correctionを支持するほど強くない。
- exp281 の常時稼働residual-offset HMMはexp263固定blendより`+1.589088 ft`悪化し、worst wellも`+30.961675 ft`だった。GR尤度を常時decoderへ入れる案は閉じている。
- exp133 のbimodal ambiguity flagは対象率`56.6857%`と広く、ambiguous側がbase modelの悪化領域でもなかった。exp177のBeam ambiguity gateもreplacement先が弱く悪化した。

## 仮説

GR emission 尤度がshiftを識別できないblockでは、PF/HMM成分によるmode選択の根拠が弱い。ただしshift 0、すなわちexp226 K16 anchorが同じGR尤度で棄却されていない場合に限れば、exp263固定blendからexp226へ小さく戻すことで、大きな誤modeのtailを増やさずRMSEを改善できる。

## 機能要件

1. 親予測は保存済みexp263固定blend、shrink先は保存済みexp226 K16とし、親PF/HMM/K16を再生成しない。
2. exp226 K16予測をshift 0とした固定13 shift `[-80,-40,-20,-10,-5,-2,0,2,5,10,20,40,80] ft`を、exp280と数値互換のraw GR/typewell Gaussian emissionで採点する。
3. 非重複512行blockごとに、top1-top2 margin、正規化entropy、shift 0 rank、best-vs-zero gap、raw observed GR shareを計算する。
4. 「GRが弱い」はouter-train-onlyの低marginかつ高entropyで定義し、絶対log-likelihood単独では定義しない。
5. exp226 admissibilityは`zero_rank<=3`または`zero_gap`がouter-train下位20%で定義する。GRがshift 0を明確に否定するblockでは発火させない。
6. raw observed GR share `>=0.80`かつ`md_since_last_known>=250 ft`の行だけを対象にする。
7. 補正は固定`alpha=0.25`、1行あたり最大`10 ft`のbounded soft shrinkとする。hard replacementやtop1 shift commitは行わない。
8. target-free score、gate、候補予測、契約をcontent SHAでfreezeした後だけsuffix true TVTを結合する。
9. real gateと、well内でblock gateを非zero circular shiftしたmatched negative controlを同じ発火量で比較する。

## 非機能要件

- Route: `pf_beam`。
- 1 readout candidate、5 fold strata、LightGBM config 0、trained fold 0、booster 0、HMM/PF/Beam/K16再生成0。
- Kaggle CPU、GPU/TPU/internet offを想定する。現時点では実行しない。
- exp263 cache manifest、exp226 OOF、raw train/typewell、hidden-like assignmentのidentityとSHAをhard guardにする。
- threshold、shift bank、block size、alpha、clip、scope、control、pass/fail基準はtruth read前に固定し、同じOOFで救済調整しない。
- gzip生成物はraw SHAとdecompressed content SHAを分け、主証拠はdecompressed content SHAとする。

## 受け入れ基準

### 今回の設計完了

- `KAGGLE_DIRECTION.md`にexp322をP1・0-model・設計確定・未実装として記録している。
- `.steering/20260721-exp322-gr-likelihood-weak-exp226-soft-shrink-readout/`に仮説、式、truth境界、固定値、停止条件を記録している。
- `experiments/exp322_gr_likelihood_weak_exp226_soft_shrink_readout/`のconfig、README、SESSION_NOTES、result、metricsをdesign-only状態で整合させている。
- `experiment_summary.md`へ未実行実験として登録している。
- Notebookと実験ロジックを実装していない。

### 今回の実装完了

- exp263 cache / exp226 OOF / raw horizontal/typewell / hidden-like assignmentをSHA hard guard付きで解決する。
- exp280数値互換の13-shift Gaussian raw-GR scoreとH512 block特徴をtarget-freeで生成する。
- outer-train threshold、shift 0 admissibility、raw GR coverage、near veto、fixed bounded shrink、well内circular controlを実装する。
- target-free score/gate/predictionをschema/content SHAでfreezeし、別関数でのみtruthをlate joinする。
- fixed technical/scientific decisionをfold/scope/by-well metricsから機械判定する。
- canonical Notebookを上書きせず、compact self-contained候補Notebookを別名で生成する。
- unit test、Jupytext round-trip、構文、ruff、strict experiment validationを通す。
- Kaggle package/push/run、inference、submissionは行わない。

### 将来の科学的PASS

- exp263固定blend parityの最大絶対差が`1e-5 ft`以下。
- changed row率が`1%--25%`、changed wellsが50以上、5 folds中4 folds以上にchanged rowがある。
- overall RMSEがexp263 `8.238331`から`0.02 ft`以上改善し、4/5 folds以上で改善する。
- activated subset RMSEが`0.10 ft`以上改善する。
- near `0--250 ft`はbitwise不変、1000+、hidden-like spatial、hidden-like typewell-purged、by-well delta p95、worst wellはいずれも悪化しない。
- real gateのoverall gainがcircular-shift controlのgainを`0.02 ft`以上上回る。

## 対象外

- 絶対GR尤度だけによる弱区間判定、exp133のambiguous flag再利用、final HMM posteriorからのgate作成。
- exp226へのhard replacement、top1 shift correction、HMM/PF/Beamの再decode、blend weight/alpha/quantile/block/clip grid。
- ML selector、truth/error/oracle gate、same-OOF rescue、inference、submission。
- exp263、exp226、exp280のcontrol再実行。
