# exp345_exp209_time_varying_gr_affine_calibration_hmm

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 technical PASS / scientific FAIL / `stage_0_full_failed_closed`
- CV / Public LB / Private LB: - / - / -
- 親実験: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- Kaggle kernel: `kentookumura/exp345-gr-affine-hmm-runtime-microbenchmark` version 2 `COMPLETE`
- Stage 1 / inference / submission: 未実行、無効

## 仮説と単一変更

閉鎖済みexp328のcausal GR affine仮説を、信頼できるexp209へ直接つなぎ直した独立再検証である。frozen exp209 base pathからcurrent-well causal `a_t,b_t` scheduleを一度だけ推定・凍結し、exp209 HMMのGR observation centerだけをidentityから置換した。

exp209 zero-fill std、missing weight 1、GR補間、Gaussian emission、41 rate states、`sig_r=0.002`、`sig_p=0.02`、position floor、momentum、prior、posterior meanは固定した。exp338とは同じ親を持つ独立兄弟で、相互依存はない。

## 検証方針

1. stable SHA順32 wellsでparent / variant各32のruntime microbenchmarkを行い、8.5時間以内に全773 wellsを処理できることを確認する。
2. runtime PASSと別承認後だけ、last-640 prefix maskの全773 wellsでparent / variantをmatched比較する。
3. technical、overall改善、4/5 folds、GR NLL、boundary、hidden-like 2 scope、worst wellの全gateをAND条件にする。
4. Stage 0全gate PASSとさらに別承認がある場合だけStage 1へ進み、FAIL時は救済なしで閉じる。

## 実行結果

Kaggle CPU version 2でlast-640 prefix maskの全773 wellsを評価し、parent 773 + variant 773 = 1,546/1,546 HMM runsを完了した。

- technical: 494,720 finite rows、fallback 0%、posterior正規化誤差`3.22e-15`、runtime 4,871.013秒（1.3531時間）で全PASS。
- overall: parent RMSE `14.501048`、candidate `14.331543`、`+0.169505 ft`改善。
- fold: 4/5改善。
- GR predictive NLL: `4.651670 → 4.646152`で改善。
- boundary jump p95: `0.010089 sigma`でPASS。
- hidden-like: 必須2 scopeのreadoutがなくFAIL。
- worst well: `+9.354827 ft`で上限`+0.25 ft`を超えFAIL。
- well勝敗: 373改善 / 400悪化。

scientific AND gateはFAILし、判定は`stage_failed_close_without_rescue`。pooled改善は確認できたが、well横断の安全性とhidden-like証拠を満たさない。

## 所見

- 良かった点: technical contract、pooled改善、4/5 folds、GR NLL、boundaryはPASSした。
- 悪かった点: 400/773 wellsが悪化し、worst wellは`+9.354827 ft`。必須hidden-like 2 scopeのreadoutもなかった。
- 解釈: 一部wellの大幅改善でpooled RMSEは良くなったが、未知wellへ安全に適用できる一貫性と証拠がない。
- 判断: 欠落証拠をPASS扱いせず、事前契約どおり同familyを閉じる。

## 実行入口と証拠

- 正規train notebook: `exp345_exp209_time_varying_gr_affine_calibration_hmm_train.ipynb`
- Jupytext source: `exp345_exp209_time_varying_gr_affine_calibration_hmm_compact_selfcontained_train.py`
- Stage 0 gate SHA256: `39296d1b900463c27f1fd65fbaa265e3c1a3a6b9d42afd9322eb03ac6140525a`
- prediction decompressed SHA256: `f2ff65b78a66c88e9993f2c362fbd9db445061670980cfffccf449ef81d4bfbc`
- affine schedule decompressed SHA256: `51827246e6b7154ff39d3d6a8c07d1bd0dd43715090b9f11036b67960d9bf0f0`

詳細なgate内訳、fold値、SHA、解釈は`result.md`と`metrics.json`を正とする。

## 次

Stage 1、post-hoc rescue、inference、submissionへ進めず、本familyを閉じる。再訪には独立根拠、別実験の事前設計、ユーザー確認が必要である。
