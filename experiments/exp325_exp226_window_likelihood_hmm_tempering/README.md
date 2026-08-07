# exp325_exp226_window_likelihood_hmm_tempering

## 状態

- Route: `pf_beam`
- 状態: 閉鎖済み・未実装・未実行
- 親: `exp323_time_varying_exp226_dip_rate_prior`

親exp323がterminal closeしたため、2026-07-22に本実験も閉鎖した。新exp325相当は、exp338 PASS後に作る新exp323相当がさらにPASSした場合だけ、新番号で設計する。

## 仮説

exp226の500行window GR scoreは、行単位Gaussian GRが区別できない繰り返し層序を識別できる。補正TVTを作るのではなく、HMM位置state上の疎な観測因子として使えばbranch誤吸着を減らせる。

windowは500行、stride 125、finite GR 50%以上。correlation/MSE/level scoreをstate方向に標準化し、overlap係数`125/500`とexp226由来posterior SD shrinkで`lambda_t`を固定する。factorはwindow中心だけに追加し、行単位GR emissionは変更しない。

## exp321との違い

exp321はZ-only経路に`±4 ft`のwindow補正を出す。exp325は補正値もexp226 predictionも作らず、exact HMM posteriorの観測尤度だけを変更する。

## 段階

- Stage 0: shift-bank MRR/top3、shuffle差、1000+、hidden-like、coverageを0 HMMで監査。
- Stage 1: 全gate PASSと別承認後だけ1 variant / 773 HMM runs。

## 検証方針

- Stage 0: 親比でMRR/top3を各0.01以上、4/5 folds、real scoreがshuffleを5/5 foldsで上回り、eligible window 25%以上、tail非悪化を要求する。
- Stage 1: 保存済みexp323親HMM比0.05 ft以上、4/5 folds、1000+・hidden-like・p95・worst非悪化を要求する。
- window identity、score surface、posterior SD、`lambda_t`、shuffle seedをsuffix truth結合前に凍結する。

## 所見

exp226の結果を混ぜるのではなく、window evidenceをHMMの生成モデルへ疎な観測因子として追加する設計である。反復層序での識別力と過剰countingの両方をStage 0で先に検査し、現在は結果なしとする。

本実験の実装、Kaggle実行、inference、submissionは今後行わない。
