# 設計

## 目的と位置付け

exp296はexp223 self-GR descriptor motifの再設計ではなく、candidate-state supportを1条件だけ追加するisolated ablationである。exp223はexp072 `likpf_mean`をRMSE 11.594897668から11.349950650へ改善した一方、worst-wellは`+46.954683 ft`悪化した。現実装はself-GR surfaceをHMM全gridへ作り、candidate stateがvisible-prefix TVT range外かをhard判定していない。

exp225はknown TVT range外をneutralにした先行実験だが、同時にexp223 descriptor motifをstate-known `TVT_input -> GR`曲線へ置換し、RMSE 14.212954500へ悪化した。したがってexp225はsupport gate単独の反証ではない。exp296ではexp223 motifを完全固定し、support maskだけを追加する。

## 固定する親実装

parent candidateはexp223 `hmm_selfgr_boost_only_a070_c100`とする。

- HMM: `step=0.35`, `n_rates=41`, `rate_span=0.10`, `sig_r=0.002`, `sig_p=0.02`, Gaussian emission, `start_sig=0.75`, `r0_sig=0.01`, `band_pad=100`, `rate_center=zero`。
- self-GR: radius 12 rows、offsets `[-12,-8,-4,0,4,8,12]`、top-k 5、stride 3、max anchors 128、last anchors 32、minimum anchors 12、window missing rate最大0.35。
- surface: matched anchor TVT中心のGaussian mixture、`sigma_tvt=12`、distance temperature 1.5、surface quadratic clip 60、exp223と同じfull-grid centering/scale。
- quality: anchor coverage、receiver window missingness、sharpness、top1-top2 gap、Type Well peak agreementをexp223のまま使う。
- final boost: `alpha=0.07`、`clip=1.0`、`boost_only`。
- raw GR interpolation、Type Well emission、prefix stats、grid、transition、posterior mean readoutを変更しない。

## 単一変更の数式

well `w`のvisible prefixから次を固定する。

\[
L_w=\min\{TVT\_input_i\mid TVT\_input_i\text{ is finite}\},\quad
U_w=\max\{TVT\_input_i\mid TVT\_input_i\text{ is finite}\}
\]

candidate state `grid[j]`のsupport maskは次のinclusiveなbooleanとする。

\[
M_w(j)=\mathbf{1}[L_w \le grid[j] \le U_w]
\]

exp223と同じ処理で得たcentered self-GR surfaceを`S_exp223(row,j)`、row qualityを`Q_exp223(row)`とすると、変更後の追加emissionは次とする。

\[
B_{296}(row,j)=M_w(j)\cdot clip(S_{exp223}(row,j),0,1)
\]

\[
logL_{296}=logL_{typewell,exp223}+0.07\cdot Q_{exp223}(row)\cdot B_{296}(row,j)
\]

maskはexp223のfull-grid centering/normalizationとpositive clipの後に適用する。これにより`M=1`のstateではboostがexp223とbitwise一致し、`M=0`では厳密に`0.0`になる。support内だけで再centeringするとinside-support値も変わるため禁止する。

finite `TVT_input`がない場合、または有限なmin/maxを構成できない場合はmaskをall-falseとし、self-GRを完全neutralにする。base Type Well HMMは通常どおり動かす。

## 非循環性と意味

最終posterior mean TVTはgate入力に使わない。gateはdecode前のcandidate stateに適用するため循環しない。posteriorはsupport外stateも選択できるが、そのstateではself-GRが寄与せず、Type Well emissionとtransitionだけで判断する。

「known TVTに含まれる」は今回`[known_tvt_min, known_tvt_max]`のinclusive rangeと定義する。nearest-known distance、局所gap、matched-anchor hullへ変更する場合は別仮説であり、exp296内の救済候補にしない。

## 入力とleakage境界

- prediction freeze前に使用可: raw horizontal `MD/Z/GR/TVT_input`、対応Type Well `TVT/GR`、well identity。
- support maskはfinite `TVT_input`だけで構築し、GR、true `TVT`、予測、errorを使わない。
- unknown suffix true `TVT`はsupport mask、surface、quality、HMM、variant selectionへ一切入力せず、prediction/support-mask SHA freeze後のmetricsだけにjoinする。
- exp115 hidden-like assignmentはmetric readoutだけに使う。
- saved exp223 controlはrow identityとdecompressed SHAを照合してから比較する。

## 実装時contract tests

- support boundary: `grid == known_tvt_min/max`はTrue、その外側1 stateはFalse。
- outside support: self-GR contribution最大絶対値`0.0`。
- inside support: mask前exp223 boostとの最大絶対差`0.0`。
- no known TVT: all-false mask、self-GR完全neutral、base HMM finite。
- no circular gate: prediction/true TVTをsupport helperの引数・入力列に含めない。
- parent parity: gateをall-trueにしたsynthetic/representative-well runがexp223出力と許容差`1e-6 ft`以内。
- fixed contract: exp223 HMM/self-GR config、active variant count、control SHA、run flagsをassertする。

## 検証設計

- score rows: train 773 wellsのofficial unknown suffix全row。
- control: 保存済みexp223 `hmm_selfgr_boost_only_a070_c100`、RMSE 11.349950650072172。
- variant: `hmm_selfgr_boost_only_a070_c100_known_tvt_support_gate` 1本。
- reporting folds: stable SHA256 well hash modulo 5。学習foldは0。
- metric: pooled/fold/by-well RMSE、MAE、within10、distance bucket、1000+、exp115 hidden-like 2面、true TVT inside/outside known range、support rate、posterior outside-support mass、step delta。
- prediction、support mask、metric schema、control joinをSHA freezeしてからtrue TVTをreadoutへ接続する。

## Hard gate

technical gateはすべて必須とする。

- input wells 773、saved-control row identity完全一致、finite prediction coverage 1.0。
- saved exp223 decompressed SHA `0eb48b5516276b0ab7b2191a52a39ebb89d9997363cc7839ede519c7863baa0c`一致。
- support外self-GR contribution最大絶対値`0.0`。
- support内boostのexp223比最大絶対差`0.0`。
- base Type Well emission / grid / transition / alpha / clip / descriptor config parity。
- support/variant prediction freeze前のunknown-suffix true TVT access 0。
- LightGBM config / trained fold / booster / parent-control retraining `0 / 0 / 0 / 0`。

performance gateはすべて必須とする。

- pooled RMSE delta vs exp223 `<= -0.05 ft`。
- reporting foldsでexp223を改善するfold `>=4/5`。
- true TVTがknown range外のscopeでRMSE delta vs exp223 `<= -0.10 ft`。
- true TVTがknown range内のscopeでRMSE delta `<= +0.02 ft`。
- distance `1000_plus`、hidden-like spatial、hidden-like typewell-purgedの各delta `<= +0.02 ft`。
- by-well RMSE p95 delta `<=0.0 ft`、最大well regression `<=+0.25 ft`。
- finite coverage 1.0、step-delta p99はexp223比非悪化。

1条件でもFAILならbranchを救済gridなしで閉じる。全PASSしてもexp209 HMM/likPF blend 10.269696以下でなければscientific supportに限定し、raw-test inferenceへ自動昇格しない。

## 実行規模と段階

別名compact self-contained train source/Notebookとcontract testsを実装し、ユーザー承認後に正規train Notebookへ採用した。

Stage 0はCPUで新variant 1本だけを773 wellsへ実行し、合計773 HMM well-runs、LightGBM config / trained fold / booster `0 / 0 / 0`、GPU 0、control再実行0を16,667.265秒で完走した。technical 12/12 PASS、performance 2/10 PASS、pooled RMSE delta `+0.809806 ft`によりFAIL-closeした。

Stage 0がFAILしたためpromotion条件の検討へ進めず、inference/submissionは実装・実行しない。

## 実験範囲

- 対象実験: `exp296_exp223_self_gr_known_tvt_support_gate`
- Route: `ensemble`
- 親実験: `exp223_joint_typewell_self_gr_hmm_likelihood_probe`
- negative/reference: `exp225_state_known_tvt_self_gr_hmm_emission`
- 変更する変数: exp223 positive self-GR boostへknown TVT candidate-state support maskを1つ追加する。
- 固定する変数: exp223 HMM、descriptor motif、quality、surface、alpha/clip/mode、入力、score rows、readout。
- 実装・実行済み範囲: compact self-contained train source/Notebook、truth-late readout、SHA/manifest、hard-gate評価、contract tests、承認後の正規train採用とKaggle CPU Stage 0。inference/submissionは対象外。

## 再現性設計

- seed policy: HMM/self-GR/support gateはRNGを使わない。reporting foldだけstable SHA256 well hashを使う。
- stochastic処理: なし。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: outer workers 2、Numba threads 2を将来固定する。well schedulingはsupport identityへ影響しない。
- runtime: CPU-only、GPU/internet無効。deterministic submission anchorではなくtrain-side no-training auditとして扱う。
- SHA: pushed source/config、raw input manifest、saved exp223 control、support mask、prediction、metrics、schemaを記録する。
- gzip: decompressed content SHAを主証拠、raw gzip SHAを補助証拠にする。
- model/submission: 学習modelとsubmissionは存在しないことをmanifestへ明記する。
- Kaggle bootstrap: loose/package/bootstrap内config/source SHA、CPU metadata、canonical id/titleをpush前に照合する。

## リスク

- parent parity: support mask以外のexp223処理を意図せず変更するとisolated ablationでなくなる。inside-support boost parityとall-true gate parityを必須にする。
- leakage: true TVTを使ってstate supportやrange paddingを選ぶ危険。support manifest/prediction freeze前のtruth accessを0にする。
- min-max bridge: known TVT range内の局所的なcoverage holeもsupported扱いになる。今回はユーザー指定のmin/maxだけを検証し、hole-aware gateへposthoc変更しない。
- global coupling: outside-state boostを消すとforward-backward全体が変わり、inside-range true rowsも悪化し得る。inside-scopeとworst-well guardで止める。
- runtime: 773 HMM decodesで5-6時間見込み。controlを再実行せず1 variantに限定する。
- CV/LB: train-side positiveでもpublic 3 wellsへ転移しない。exp209 blend以下と別承認なしにinferenceへ進めない。
- reproducibility: Numba/parallel浮動小数差があり得るためSHAとmetric toleranceを分けて記録する。
