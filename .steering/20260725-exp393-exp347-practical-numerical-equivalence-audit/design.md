# 設計

## アプローチ

exp347のFAILはposterior cell単位の最大絶対差だけに基づき、最終readoutであるposterior mean TVT差を保存していなかった。exp393では同じtemporary neural modelから生成したunaryをdetach/freezeし、scalarとbatchedの両実装へ同一tensorとして渡す。これによりmodel stochasticityとnumerical kernel差を分離する。

scalar FP32を基準に、batched codeをbatch size 1で実行して「batch次元を追加しただけの差」、production batch size 4で実行して「inter-window並列化とpaddingを含む差」を測る。固定先頭4 windowsではscalar FP64 forward/posteriorを追加し、両FP32経路が高精度参照からどの方向へずれるかを診断する。FP64は原因帰属用で、practical gateの比較基準を実行後に変更しない。

## Stage 0測定

### 固定入力

- exp347 fixed16 window/boundary manifestと同じselection algorithm、seed 42、fold 0 outer-train wells。
- exp347と同じtemporary neural unary architecture、前処理、state grid、41 rates。
- unaryは1回だけ生成し、content SHAを保存してから各modeで共有する。
- truthはwindow selectionやunary生成に使わず、exp347と同じteacher boundary/loss parityと最終集計にだけ使う。

### 比較mode

1. `scalar_fp32_reference`: exp347 scalar exact forward-backward。
2. `batched_fp32_batch1`: batched実装を1 active window、paddingなし相当で実行。
3. `batched_fp32_batch4_production`: 4-window padded/masked production実装。
4. `scalar_fp64_diagnostic`: 先頭4 windowsだけの高精度forward/posterior参照。

FP64診断のためdtype hard-codeを一般化する場合も、production FP32 pathの演算式、mask、transition、gridを変更しない。scalar/batched FP32の実行結果を先にfreezeしてからFP64診断を行う。

### 保存指標

- posterior cell max / mean / p99 abs error。
- posterior mean TVT差のRMSE / mean abs / p95 / p99 / max。
- marginal MAP state一致率、disagreement row数、state距離、top-2 probability margin。
- row-wise total variation distanceのmean / p95 / p99 / max。
- posterior row sum誤差、finite率、invalid mask値。
- structured loss、partition、unary gradient、AdamW 1-step update差。
- batch-1、batch-4、FP64参照別のruntime、peak memory。
- window/boundary/padding/unary/comparison manifestと全SHA。

## Gateと段階

Stage 0はrequirementsのpractical gateをAND評価する。PASSしても「exp347がPASSだった」と書き換えず、`exp393 practical equivalence PASS`としてのみ記録する。

PASSかつ別ユーザー承認後だけStage A fold 0の1 neural modelを候補とする。Stage Aはexp347で未実行だった科学gateを継承し、real GRがshuffle/geometry/saved exp209より改善すること、tail非悪化、runtime/memoryを要求する。Stage A controlは保存済みexp209/exp221を参照し、親/controlをGPU再学習しない。

Stage 0 FAIL時はTVT差閾値、MAP一致率、dtype、batch size、padding、compile/fused kernel、window選択を救済せずcloseする。原因診断は保存するが同じrun内でgateを選び直さない。

## 実験範囲

- 対象実験: `exp393_exp347_practical_numerical_equivalence_audit`
- Route: `ensemble`
- 親実験: `exp347_prefix_gr_unary_batched_window_exact_ssm`
- 変更する変数: promotion判定をposterior cell max errorから、最終posterior mean TVTとMAPを中心とする事前固定practical equivalence gateへ変更する。
- 固定する変数: raw data、fold、fixed16 windows、unary model、preprocessing、objective、boundary、optimizer、state grammar、batch-4 production path、decoder、science gate。
- 完了範囲: Stage 0のFAILを履歴保持したユーザーoverrideによるStage A fold 0の
  1 neural modelをKaggle T4 version 4で完了した。Stage A gateはFAILし、
  Stage B、推論、提出なしでbranch close。

## 再現性設計

- seed policy: seed 42 + well/window/mode keyからstable SHA256を生成する。global RNGを並列処理内で共有しない。
- stochastic処理: 将来のtemporary neural model初期化とCUDA convolution。比較前にmodel eval、dropout off、unaryをfreezeし、比較mode間のstochasticityを除く。
- PF/Beam / likelihood-PF / seed bagging: なし。
- 並列処理: dataloader worker 0。mode間を並列実行せず、同じfrozen unaryを順番に読む。
- runtime: Kaggle T4、internet off、CuDNN benchmark false、deterministic algorithms warn-only。GPUのbitwise一致は主張しない。
- SHA: raw input manifest、exp347 source/config/report、fold/window/boundary/padding、frozen unary、各mode posterior/readout、comparison report、package notebook/config/metadata、kernel version/logを記録する。
- Kaggle bootstrap: prepare後にmetadataと埋込configのselected stage、T4、internet off、kernel sourceを照合する。
- model/prediction/submission: Stage 0はpersisted model/prediction/submissionを生成しない。Stage A以降は別承認時にmodel manifestとprediction SHAを追加設計する。
- deterministic anchor: false。別version rerunのunary/comparison SHAが一致または説明可能になるまでanchor化しない。

## リスク

- posterior meanが一致しても、長い学習で微小gradient差が蓄積する可能性がある。Stage 0 PASSはStage A成功を保証しない。
- MAPはnear-tie rowで微小差に敏感で、TVT meanが同じでも一致率gateをFAILし得る。
- FP64参照はT4で遅く、4 windowsへ限定してもruntimeを押し上げる可能性がある。
- batch-1とbatch-4の差がpaddingではなくGPU reduction kernel選択に依存し、rerunやCUDA versionで揺れる可能性がある。
- fixed16 windowsはfull-well distributionを完全には代表しない。Stage Aではfull-well decodeを別途評価する。
- 直近のKaggle GPU quota不足により実行待ちになる可能性がある。実行環境を変えると比較条件が変わるため別承認を要する。

## Stage A override設計

- 分岐条件はStage 0 PASSではなく、3 FAILを明記した
  `execution.stage_a_user_override`の明示承認と固定Stage 0 report SHAに置き換える。
- overrideはStage Aへの進行だけを許可し、Stage 0 gate、exp347判定、Stage A science
  gateを変更しない。
- exp347 compactのfour-window training、outer-train early stopping、freeze-first
  outer-valid decode、saved exp209比較、hidden-like readoutをexp393正規Notebookへ追加する。
- GPU costは1 variant / 1 architecture / fold 0 / seed 42 / neural model 1 /
  booster 0 / control再学習0。保守的runtime見積りは`5.108737 h`。
- Stage A FAIL時は救済せずclose。PASS時もStage Bはさらに別承認とする。
