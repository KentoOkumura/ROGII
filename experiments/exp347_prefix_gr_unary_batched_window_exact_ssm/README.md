# exp347_prefix_gr_unary_batched_window_exact_ssm

## 状態

- Route: `ensemble`
- 状態: Stage 0 technical parity FAIL、terminal close
- 優先度: 完了。exp348の先行条件は充足
- 親: terminal closedの`exp332_prefix_gr_unary_fixed_window_structured_ssm`

## 仮説

exp332の4 windows分のexact structured lossを1件ずつ計算せず、4件を同時にGPUへ載せれば、科学契約と実効optimizer batchを変えずにT4 fold runtimeを`8.5 h`以内へ短縮できる。

## 単一変更

- exp332: `1 window/batch × gradient accumulation 4`
- exp347: `4 windows/batch × accumulation 1`
- objective、window、teacher boundary、architecture、state grammar、controls、full-well decodeは固定する。

## 検証方針

固定4 windowsでscalar/batchのloss、posterior、gradient、optimizer 1-step parityを確認した後、固定16-window T4 Stage 0で保守的fold runtime、peak memory、exp332比speedupを測る。すべてPASSしない限りStage Aへ進まない。

## Stage 0実行

- 固定16 windows、benchmark variant 1、一時model 1。
- 永続model、trained fold、LightGBM config、booster、PF/Beam、control再学習はすべて0。
- scalar/batch parity、保守的runtime`<=8.5 h`、peak`<=14 GB`、exp332比`>=1.55x`をAND gateにする。

Kaggle T4 version 1（id_no `128239400`）を完了した。保守的fold外挿`5.108737 h`、speedup`2.574244x`、peak`5.928168 GB`はPASS。一方posterior max abs errorは`1.4662743e-5`で固定上限`1e-6`をFAILした。loss/partition/update差0、gradient差`1.4319085e-8`、padding/finiteはPASSしたが、AND gate不通過としてbranchを閉じた。

## 結果

- Compute gate: PASS。
- Technical parity: posteriorのみFAIL。
- 総合Stage 0: FAIL、`close_without_batch_or_science_rescue`。
- Stage A model / prediction / submission: すべて未生成。
- reportとmanifestのSHAは取得ファイルと一致。

## 実装境界

compact self-contained train候補を正規train Notebookへ採用した。正規inferenceはscaffold placeholderのまま維持している。trainには次を実装した。

- position/rate/row/inactive-window paddingを明示maskする4-window exact forward-backward。
- per-window valid-row正規化lossの算術平均と、1 batch = 1 optimizer update。
- scalar/batch loss・partition・posterior・gradient・AdamW 1-step parity report。
- fixed 16-window Stage 0のbatched train/forward-only計測と、stable length順4-well・3-control full-well decode。
- Stage A fold 0学習とfreeze-first outer-valid batched decode。Stage 0 PASSと別承認なしには到達できない。

Kaggle Stage 0 package/push/runと証拠回収まで完了した。Stage A model、prediction、submissionは生成していない。

## 実装検証

- Jupytext変換 / `--test`: PASS。
- py_compile / Ruff `E,F,I,UP,B`: PASS。
- 専用pytest: `16 passed, 2 skipped`。skipはローカル環境にPyTorchがなく、torch scalar/batch numerical testを実行できないため。
- `make validate-exp`: strict PASS。

## 所見

4-window batch化は計算高速化には有効だったが、exact posteriorの事前固定等価性を満たさなかった。事前契約どおりbatch/padding/compile/fused kernel/閾値/科学契約の救済を行わず、Stage A/B/C、推論、提出へ進まない。

## 次

exp347はterminal closeとして維持する。独立仮説のexp348は先行条件を満たしたが、実装・実行は別判断とする。

## 文書

- Steering: `../../docs/legacy/steering/20260722-exp347-prefix-gr-unary-batched-window-exact-ssm/`
- 設定: `config.yaml`
