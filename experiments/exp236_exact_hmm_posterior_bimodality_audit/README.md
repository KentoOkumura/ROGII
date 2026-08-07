# exp236_exact_hmm_posterior_bimodality_audit

## 状態

Kaggle CPU train-side audit v1 は完了。route は `ensemble` だが、既存 exp148 OOF center と
固定 exact HMM posterior を診断しただけで、新規学習・推論・提出は行っていない。

## 仮説

exp221 の posterior mean が二峰 posterior の谷に落ちる、または dominant mode が頻繁に
入れ替わるなら、exp221 の CV 改善と Public LB 転移不足・worst-well 悪化を説明する手がかりに
なる。

## 検証方針

exp221 と同一の `exp148 lgb_mean` OOF center、`sigma=20.0`、`lambda=0.50`、HMM grid、
emission、transitionを固定した。well ごとに posterior をメモリ上だけで解析し、peak、mass、
valley、entropy、mode persistence を保存した。true TVT は decoder 評価と oracle top2 readout
にのみ使い、peak 選択・二峰判定・mode 追跡には使っていない。

## 所見

二峰 row は 0.9355%（138 wells、317 segments）、mean-in-valley row は 0.1792%、mode mass
switch は 17 回だった。posterior mean RMSE 8.327728486 が最良で、MAP は +0.037431949、
dominant-mode conditional mean は +0.004025866 悪化した。したがって direct decoder の変更は
不採用であり、raw-test inference / submit には進まない。

## 参照ファイル

- `config.yaml`
- `posterior_bimodality_audit.py`
- `exact_hmm_smoother.py`
- `exp236_exact_hmm_posterior_bimodality_audit_train.ipynb`
- `result.md`
- `SESSION_NOTES.md`
