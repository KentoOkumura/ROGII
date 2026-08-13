# exp295 architecture contract

## 正規契約

この文書はexp295の入力境界、数理モデル、stage分岐、成功条件を実装都合や中間結果から独自変更しないための正規契約である。変更する場合はユーザーの明示承認を得て、`config.yaml`、この文書、`KAGGLE_DIRECTION.md`を更新し、新しい実験へ分ける場合はその`requirements.md`へ契約を移す。

2026-07-19にStage A実装が承認され、別名compact self-contained train候補、fail-closed inference候補、専用contract testsを実装した。canonical train NotebookとStage A GPU pushはその後承認済み。Stage B/C、inference、submissionは未承認である。

2026-07-20のversion 2は、疎なtruth jumpをhard truth pathとして固定grammar内に置けずruntime failureした。ユーザー承認により、decoder/state grid/transitionを固定したまま、学習objectiveだけをGaussian soft-label structured likelihoodへ修復した。

## 一文でのモデル定義

対象井1本を1 sampleとして、対象井自身のhorizontal GRと対応Type Well GRからknown-prefix-conditioned row x TVT unaryを学習し、known prefixをhard clampしたfixed exact state-space posterior meanでhidden suffix TVTを推定する。

## Neighbor-free boundary

inference graphに入れてよいhorizontal sourceは対象井自身の1本だけである。

許可するもの:

- 対象井自身の`MD/X/Y/Z/GR/TVT_input`。
- 対象井に対応して提供されるType Wellの`TVT/GR` curve。
- outer-train wellsから学習済みの共有model parameters。

禁止するもの:

- 他horizontal wellのTVT、TVT_input、GR、path、candidate、prediction。
- XY近傍、same-typewell donor、neighbor copy、peer atlas。
- exp263/exp293 candidate bank、exp264 selector、既存ML/PF/Beam予測。
- validation/test wellに対するgradient update。

## Complete-well emission

horizontal/Type Well encoderは64次元multi-scale embeddingを生成する。visible prefixのtrue対応は`TVT_input`からだけ構成し、masked attention poolで32次元context`c_w`へ集約する。`c_w`はFiLM scale/biasとpositive temperatureだけを生成する。

row`i`とTVT state`q`のunaryは次とする。

\[
u_\theta(i,q;c_w)=\frac{\langle W_h(c_w)h_i, W_t(c_w)t_q\rangle}
{\tau(c_w)\|W_h(c_w)h_i\|\|W_t(c_w)t_q\|}
\]

`MD/X/Y/Z`はこのunaryへ入力せず、固定transitionと初期stateだけに使う。real GR、Type Well circular shuffle、zero-GR/geometry-onlyを同じweightsでdecodeし、GR shortcut attributionを必須監査する。

## Fixed state-space model

stateはexp209と同じTVT gridと41 rate statesで構成する。transition、process noise、start prior、bandを変更しない。visible `TVT_input`はhard clampし、hidden suffix全体をexact log-space forward-backwardする。

primary predictionは各rowのTVT posterior mean。Viterbi、MAP、posterior standard deviation、entropy、edge massは診断であり、prediction selectionに使わない。

## Training contract

- outer split: whole-well 5-fold GroupKFold。
- valid view: official `TVT_input` maskだけ。
- train views: outer-train wellsのofficial startと、可能な場合だけ256/512 rows前のdeterministic pseudo-cut。
- batch: complete well 1本、gradient accumulation 4 wells。
- loss: Gaussian soft-label structured NLL 1.0（label observation `sigma=0.35 ft`）+ local true-state CE 0.25。structured項は通常partitionと`unary + Gaussian label emission`の条件付きpartitionとの差で、勾配は通常posterior minus label-conditioned posteriorとする。
- optimizer: AdamW `3e-4`, weight decay `1e-4`, max 8 epochs, clip 1.0, AMP。
- early stopping: outer-train内stable holdoutだけ。
- test-time backprop: なし。
- architecture/loss/sigma/band/temperature grid: なし。

outer-valid truthはmodel、unary、posterior、negative controls、row identity、SHA manifestをfreezeした後にだけ別loaderで読む。

## Stage A

fold 0、architecture 1、seed 1、neural model 1。control model 0、LightGBM config/booster 0、PF/Beam run 0、parent retraining 0。

全PASSをStage B開始条件とする。

1. finite prediction 100%、target-in-grid 99.5%以上、prefix clamp差`<=1e-6 ft`。
2. real NLLがshuffleより0.05 nats/token以上良い。
3. real within10 posterior massがshuffleより0.03以上高い。
4. real RMSEがgeometry-onlyとexp209の双方を0.25 ft以上改善する。
5. exp209比well p95非悪化、worst regression 10 ft以下。
6. peak GPU memory 14 GB以下、fold runtime 8.5時間以下。

FAIL時はStage Bへ進まず、exp295内で救済しない。

## Stage B

Stage A modelを再利用し、fold 1-4の4 modelsだけを追加する。合計5 models、1 architecture、1 seed。negative controlsは同じweightsを使う。

GR attribution PASS:

- realがshuffleとgeometry-onlyの各controlをpooled RMSEで0.50 ft以上改善。
- realが各controlを5/5 foldsで改善。
- real true-state NLLがshuffleより5/5 foldsで良い。

LB 5.x promotion PASS:

- pooled OOF `<=6.0 ft`、stretch `<=5.0 ft`。
- exp221を5/5 foldsで改善。
- exp221比で1000+とhidden-like 2面をすべて改善。
- exp221比well p95非悪化、worst regression`<=5.0 ft`。
- GR attribution、finite、continuityを全PASS。

判定:

- promotion PASS: 別承認後に同じexpのStage Cへ進める。
- `6.0 < OOF <=6.75`かつGR attribution PASS: 仮説signalあり。ただしexp295 inferenceは禁止し、別expのarchitecture iterationだけを検討できる。
- それ以外: branch close。

## Stage C

Stage B promotion PASS後の別承認が必要。5 outer-fold modelsでcurrent-test posterior meanを生成する。同じarchitecture/transition/input contractを維持し、既存ML/PF/Beamとのblendやcandidate routingを追加しない。

raw-test input parity、model manifest、prediction content SHA、runtime、submit-checkを通過するまでsubmissionを作らない。提出はさらに別承認とする。

## Failure policy

次をexp295内で行わない。

- valid scoreを見たband、dilation、embedding、loss weight、label sigma、temperature、epoch、pseudo-cutの変更。version 2の学習不能を解消する承認済み単一objective修復は例外として履歴固定し、比較gridへ展開しない。
- GR control不合格後のfeature追加やtrajectory shortcut。
- candidate bank、hard top1、Viterbi replacement、posterior candidate mixture。
- pseudo-start reliability gate、well ID gate、worst-well個別rule。
- neighbor/data donor、same-typewell path、spatial prior。
- Stage B未通過でのinference、blend、submission。
