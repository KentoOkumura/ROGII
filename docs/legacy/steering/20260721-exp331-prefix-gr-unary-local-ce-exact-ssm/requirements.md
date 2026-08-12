# 要件

## 依頼

exp295でKaggle timeoutしたcomplete-well GR unary仮説を、各rowの正解Type Well TVT stateに対するlocal Cross Entropy（CE）だけで学習し、fixed exp209 exact SSMは評価・推論時だけ使う別実験として設計する。今回はbacklog、steering、実験scaffoldと設計確定までとし、実装、Notebook実行、Kaggle push、推論、提出は行わない。

## 2026-07-21 実装承認

設計確定後、ユーザーの「exp331を実装してください」をStage 0/Stage Aコードの実装承認として受領した。承認範囲は別名compact self-contained train候補、fail-closed inference候補、専用contract tests、設定・記録更新までとする。canonical Notebook上書き、Kaggle package/push、T4 microbenchmark実行、Stage A学習、Stage B/C、推論、提出は含まない。

## 仮説

exp295の精度仮説は未評価であり、失敗原因は通常posteriorとlabel-conditioned posteriorを学習中の全viewで計算した4-sweep exact DPの計算量だった。structured gradientをlocal CEへ置き換えれば、同じGR unary architectureをKaggle時間内に学習でき、学習後だけfixed exact SSMでcomplete-well posterior meanを計算することで、GR alignment仮説を反証可能にできる。

## 制約

- Route: `ensemble`。学習はneural unaryだが、最終TVTはneural unaryとfixed exact SSMの双方が本質的に寄与する。
- 親実験: `exp295_prefix_anchored_wholewell_gr_alignment_ssm`。
- exp295のinput allowlist、fold map、architecture、preprocessing、exp209 state grid/transition、posterior-mean readout、controls、promotion gateを固定する。
- 学習lossはnearest true Type Well TVT stateへのhard local CE `1.0`のみ。structured NLL、SSM forward-backward、transition lossをoptimizer graphへ入れない。
- early stoppingもstable outer-train holdoutのlocal CEだけを使い、outer-valid truthやexact SSM scoreでepochを選ばない。
- exact SSMはmodel freeze後のfold評価・controls・承認後の推論だけで使う。
- outer-valid suffix TVTはmodel、unary、posterior、row/control manifestをfreezeした後だけ評価へjoinする。
- 再現性は`docs/06_reproducibility.md`に従う。
- full Stage A前に固定16-view T4 microbenchmarkを行い、fold 0学習+評価の外挿が8.5時間超ならfull Stage Aを開始しない。
- exp295 version 4、exp331内のstructured/window案、architecture/loss/band/temperature/view/epoch grid、parent/control再学習は禁止する。

## 受け入れ基準

- `docs/legacy/steering/20260721-exp331-prefix-gr-unary-local-ce-exact-ssm/`に目的、数理契約、入力境界、計算量gate、Stage A/B/C、成功/失敗条件が固定されている。
- `experiments/exp331_prefix_gr_unary_local_ce_exact_ssm/`にcompact self-contained train/inference候補、config、専用contract tests、実装記録があり、未実行であることが明記されている。
- Stage Aは`1 architecture × fold 0 × seed 42 = 1 neural model`、LightGBM/PF/Beam/parent-control再学習0と固定されている。
- Stage BはStage A全PASSと別承認後だけfold 1--4の4 modelsを追加し、fold 0を再学習しない。
- Stage CはStage B promotion PASSと別承認後だけ同じexp内で実装する。
- feature/model/prediction/submission SHAとKaggle kernel versionの記録方針がある。
- training lossとearly stoppingからexact SSM/structured objectiveが除かれ、local CEだけを呼ぶcontract testがある。
- 固定16 viewsをsuffix長quartileごと4件選ぶStage 0と、保守的runtime/peak-memory gateが実装されている。
- Stage 0 PASS記録と別承認なしにStage Aを実行できず、Stage B promotionと別承認なしにinferenceを実行できない。
- `make validate-exp EXP=exp331_prefix_gr_unary_local_ce_exact_ssm`がstrictで通る。
