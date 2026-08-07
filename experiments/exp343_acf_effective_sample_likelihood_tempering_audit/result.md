# exp343 結果

## 結論

Kaggle private CPU version 1でStage 0を完了し、固定AND gateをFAILしたため
`stage_0_failed_close_without_rescue`でbranchを閉じた。well別`tau_eff`を
安定したtempering係数として識別できず、Stage 1は不適格である。

- Kernel: `kentookumura/exp343-acf-effective-sample-tempering-train`
- Kaggle id / version: `128358348` / `1`
- 実行時間: `273.66704466799996 sec`
- 実行環境: Kaggle CPU、internet/GPU/TPU off
- 実行量: diagnostic 1、reporting folds 5
- HMM well-run / model config / trained fold / booster / 親control再実行:
  `0 / 0 / 0 / 0 / 0`

## Stage 0 gate

| 項目 | 実測 | 固定条件 | 判定 |
| --- | ---: | ---: | --- |
| well数 | 773 / 773 | 773 / 773 | PASS |
| joint-evaluable | 295 / 773 = 0.381630 | 0.90以上 | FAIL |
| fallback | 478 / 773 = 0.618370 | 0.10以下 | FAIL |
| full/last-512 Spearman | undefined | 0.70以上 | FAIL |
| median absolute log ratio | 0.0 | log(1.5)以下 | PASS |
| stable folds | 0 / 5 | 4 / 5以上 | FAIL |
| pooled median `tau_eff` | full 4.0 / tail 4.0 | 1.25以上 | PASS |
| upper-clip率 | full 0.997413 / tail 1.0 | 0.25以下 | FAIL |
| fold median `tau_eff`比 | full 1.0 / tail 1.0 | 1.50以下 | PASS |

fold別joint-evaluable率は`0.357143--0.402597`、fallback率は
`0.597403--0.642857`で、5 foldすべてunstableだった。

## 解釈

outer-train foldのraw tau中央値はfullで`9.771436--10.039963`、
last-512で`24.258286--25.172847`だった。固定clip上限4を大幅に超え、
fullは99.74%、tailは100%のwellで`tau_eff=4`へ潰れた。このため
Spearmanは定数列となって未定義であり、log ratio 0.0も安定性の証拠ではなく
clipの結果である。

加えて61.84%のwellがfallbackした。したがって「GR residualに強い系列相関がある」
ことは再確認できても、current-well固有のtempering係数をこの固定契約で安定推定
できたとは判断しない。`tau_eff=4`を妥当な一律温度として採用する根拠にも使わない。

事前契約どおりlag、support、clip、temperature、downsamplingの救済gridは行わない。
Stage 1 HMM、prediction、inference、submissionは未実装・未実施のままである。

## 再現性

- scientific contract content SHA:
  `59faf92ce0130ec025fb5d244dfa35170aa338b56e77e87baba593fa071177a0`
- known-prefix residual content SHA:
  `58b9b269ab2374270463b000edb2a671197e5514dbb1dbc664cabb941cc12cea`
- ACF lag content SHA:
  `2383489b6d6feefeda71c2ae4a35122a001145a7d175c06b2a78622eecd87512`
- tau schedule content SHA:
  `42391a82f4228ecae79755050b1b589732ba8e7d319ca5f7d159aeeb8b3d7022`

確認にはsummary、gate、fold metrics、fold priors、stability readoutなどの
小さい出力だけを選択取得した。known residual、ACF、tau scheduleの大きいarchiveは、
Kaggleログと記録済みcontent SHAで十分なため取得していない。

最初の長いkernel slugへのpushはKaggle `SaveKernel 400`で科学実行前に失敗した。
式・入力・gateを変更せず短いcanonical slugへ直し、成功したversion 1だけを
科学実行として記録した。
