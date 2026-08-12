# 設計

## アプローチ

exp384はsmooth baseとpiecewise componentsをposterior平均するが、domain posteriorは
地層geometryとprefixだけで決まる。本実験では各物理pathが予測するTVTを使って
target typewell GRをhorizontal軌跡へ写像し、実測horizontal GRとの一致度を
256 ft windowごとのStudent-t emissionにする。

stateはsmooth base 1本とexp384 posterior上位最大8 component pathsである。
exact forward-backwardでwindowごとのstate posteriorを求め、primary TVTは
posterior-weighted path平均に固定する。

### 1. 先行条件と固定candidate bank

- exp383/384 Stage 0/1全PASSを必須とする。
- exp384のinput manifest、base/component path、domain posterior、
  uncertainty、row identityのcontent SHAをhard pinする。
- 各well/windowでcandidate順を`base`、続いてexp384 component ID昇順に固定する。
- 最大candidate数は9。componentが8未満でもdummy candidateを作らずavailability maskを使う。
- candidate pathは再fit、blend、offset、clipしない。

### 2. typewell GR projection

- target typewellのfinite`TVT/GR`をTVT昇順へstable sortする。
- duplicate TVTはGR medianへ集約する。
- 各candidate TVTでtypewell GRを線形補間し、typewell範囲外はmissingとする。
- horizontal GRとのresidualを作り、双方finiteのrowだけをemissionへ使う。
- sigmaは対象known prefix上の親exp384 pathについてGR residual MADから
  `1.4826*MAD`で推定し、`[5,60] GR API`へclipする。
- finite prefix pairが64未満ならouter-train LOO sigma medianへfallbackする。
- target suffix TVTはsigma推定に使わない。

### 3. fixed GR window emission

- window length 256 rows、stride 64 rowsに固定する。
- valid pairが64未満のwindowはneutral emissionとする。
- Student-t `df=4`、row NLLのvalid-count平均をwindow scoreにする。
- score clipはwindow当たり`[-25,0]` log-likelihoodへ固定する。
- candidateによりtypewell外挿となるrowはmissingで、値を補完しない。
- target-free controlとしてhorizontal GRをwell内で512 rows circular shiftした
  単一control scoreを同時に作る。shift方向は正に固定する。

### 4. state dynamicsとposterior

- state priorはexp384のwindow平均domain posteriorを使う。
- transitionは次へ固定する。

```text
P(s_t=j | s_{t-1}=i)
  = 0.98 * I(i=j) + 0.02 * prior_t(j)
```

- unavailable stateのprior/transition/emissionは0にし、残りを正規化する。
- log-space exact forward-backwardを使い、Viterbi/hard top1はprimaryにしない。
- window posteriorをoverlap-addでrow posteriorへ戻す。
- eligible windowがないrowはexp384 posteriorとpredictionへexact fallbackする。
- primary predictionはrow posteriorによるcandidate TVT平均。
- posterior entropy、top1 margin、eligible count、real-vs-shift score差を保存する。

### 5. known-prefix Stage 0 backtest

target truthを開く前に、各outer-valid wellの既知prefix内でrolling-origin backtestを行う。

- prefix長が384 rows以上のwellだけeligible。
- last 256 known rowsをheldout、以前のknown rowsだけでexp383/384 prefix校正を再fitする。
- heldoutのTVT_inputはcandidate/path/posterior freeze後の評価だけに使う。
- primaryはparent exp384とGR posterior pathのheldout RMSE差。
- real GRと512-row circular shift controlのheldout MRR/top3/entropy差を報告する。
- backtestにsuffix true TVTと生Formationを使わない。

## 実験範囲

- 対象実験: `exp385_gr_typewell_likelihood_on_vector_drift_paths`
- Route: `pf_beam`
- 親: exp384
- 変更する変数: typewell projection、GR Student-t emission、state dynamics、posterior平均。
- 固定する変数: exp383/384のfield、candidate path、prefix、fallback、component identity。
- Stage 0: 0-full-HMMのscore/backtestと16-well resource/parity audit。
- Stage 1: Stage 0全PASSと別承認後だけ773 exact forward-backward well-runs。
- fitted ML model / booster / PF particle / Beam: 0。
- parent control再実行0。

## 検証段階

### Stage 0: target-free likelihood/backtest/resource

- exp384 input/candidate SHA一致、row/key/availability finite。
- target suffix truth、生Formation、error/oracle read 0。
- candidate count`1..9`、base availability 1.0。
- eligible GR window率`>=0.25`。
- eligible well率`>=0.50`。
- posterior normalization max abs error`<=1e-12`。
- ineligible rowのexp384 parity max abs`<=1e-8 ft`。
- known-prefix heldout pooled RMSE gain vs exp384`>=0.10 ft`。
- heldout positive folds`>=4/5`。
- real-minus-circular MRR gain`>=0.01`。
- real posterior entropyがcircularより低いfold`>=4/5`。
- 16-well projected runtime`<=30,600 sec`、peak RSS`<=25 GB`。

FAILならfull decoder、emission/transition救済、late OOF scoringを行わない。

### Stage 1: full exact decoder

- exp384比pooled RMSE gain`>=0.50 ft`。
- positive folds`>=4/5`。
- 1000+ gain`>=0.50 ft`。
- hidden-like spatial/typewell-purged gainが各`>=0.25 ft`。
- GR-missing bucket delta`<=0 ft`。
- ineligible row parity max abs`<=1e-8 ft`。
- by-well tail、GR coverage、entropy/margin bucketは必須報告、初回signal gateではreport-only。

PASSしてもcurrent-test inference、submissionは別承認とする。

## 再現性設計

- seed policy: RNGなし。candidate/window/state/well順を固定する。
- stochastic処理: なし。
- decoder: log-space exact forward-backward、particle/Beamなし。
- parallelism: well単位並列後にwell/row/state順へ再整列する。
- runtime: Kaggle CPU / GPU off / internet off。
- SHA: exp384 input、candidate bank、typewell index、real/circular window scores、
  transition, posterior, OOF predictionをlogical content SHAで記録する。
- gzip: decompressed content SHAを主証拠にする。
- deterministic anchor: 初回runでは主張せず、rerun score/posterior/prediction SHA一致後に再評価。
- bootstrap: 実装承認後にconfig/source/candidate input SHA、CPU/internet/run flagsを照合する。

## リスク

- candidate diversity: exp384 pathsが近すぎるとGRで識別できない。
- GR registration: physical TVT誤差とGR位置ずれを混同する可能性があるが、
  本実験では追加offset stateを入れない。
- missingness: eligible thresholdとexact fallbackで扱い、GRを補完しない。
- typewell mismatch: target typewellがhorizontal geologyを代表しないwellがあり得る。
- posterior averaging: RMSEではhard top1より安定しやすいが、mode間平均が非物理pathになる可能性を報告する。
- CV/LB: GR/typewell品質分布がhiddenで変わるためcoverage bucketとhidden-likeを必須にする。
- post-hoc: window/stride/df/sigma/clip/transition/circular shiftを救済しない。
