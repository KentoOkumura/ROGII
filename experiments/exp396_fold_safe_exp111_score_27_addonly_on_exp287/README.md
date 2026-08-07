# exp396_fold_safe_exp111_score_27_addonly_on_exp287

## 状態

- Route: `ml_model`
- 状態: Stage B 15/15 boosters完了・promotion gate FAIL・branch閉鎖
- CV: `8.134294735`（exp287比`-0.002413486 ft`）
- Public LB: まだなし
- Private LB: まだなし
- Submit ID: なし
- 作成日: 2026-07-25
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`
- clean tail control: `exp264_exp263_candidate_confidence_dual_selector`

## 仮説

exp111のwithin10 probability / expected absolute errorから作る27列は、5つのPF/Beam系候補の
局所信頼度を表し、exp287の421特徴へfold-safeに追加すればTVT OOFを改善できる。
旧27列の問題は保存済みfold0 scorerを全trainへ適用したnon-OOF生成であり、feature family自体の
否定ではない。

## 変更点

- exp287の421特徴、fold、target、3 LightGBM config、評価契約は固定する。
- downstream outer 5 foldsの各outer-train内で4 GroupKFoldを作る。
- outer-train scoreはinner OOF、outer-valid scoreはouter-trainだけで学習した4 model平均にする。
- exp111と同じ5 candidates × 2 objectivesを使い、40 CPU scorer boostersを予定する。
- scorerごとにinner-train由来の48-column imputation medianを保存する。
- 10 score coreから固定27列を導出し、Stage Bでは421 + 27 = 448特徴だけを評価する。

## 実行段階

| Stage | 内容 | 予定量 | 現在 |
| --- | --- | ---: | --- |
| A | strict nested exp111 scorer生成・品質監査 | preflightは0 booster、学習は5 outer × 4 inner × 2目的 = 40 CPU boosters | version 2、40/40 boosters、全gate PASS |
| B | exp287 421 + score 27のTVT add-only | 1 variant × 3 configs × 5 folds = 15 GPU boosters | version 1、15/15完了、promotion FAIL |
| inference | outer別4 scorer ensemble + 保存15 TVT model | booster学習0 | gate FAILにより実装・実行しない |
| submission | 形式検証・competition submit | 未定 | 実行しない |

control boosterの再学習は0。Stage Bの15 GPU boostersは再提示後に明示承認済み。

## 検証方針

Stage Aはleakage/coverage/model・median・schema数、runtime 30,600秒、peak RSS 25GBに加え、
expected-error MAE、within10 logloss、within10 Brierがcandidate prior比でpooled改善かつ4/5 folds
改善することを全ANDで要求する。

Stage Bは以下を全ANDで要求する。

- exp287比pooled OOF delta `<= -0.02 ft`
- 4/5 folds以上がexp287以下
- near / mid / 1000+ / hidden-like各scopeのdeltaが `<= +0.02 ft`
- by-well delta p95がexp287比 `<= 0.00 ft`
- corrected exp264比worst-well delta `<= +0.25 ft`
- corrected exp264比の+1/+3/+5 ft悪化well数が `135/39/14` 以下

## 所見

Kaggle private CPU version 2で40 CPU boostersを完了し、technical 22/22、
scorer-quality 6/6、runtime/memory gateを全PASSした。続くprivate T4 version 1は15/15 boostersを
完走したが、OOF `8.134294735`はexp287比`-0.002413486 ft`にとどまり、foldは2/5のみnonworse。
scope最大悪化`+0.026155871 ft`、by-well p95 `+0.342926545 ft`、corrected exp264比worst
`+7.802733095 ft`で、固定promotion gateは1/6 PASSだった。scorer品質は高くても、
27列のdownstream価値はfold/scope/tailへ安定転移しなかったためbranchを閉じた。

## 禁止事項

- 保存済みexp111 fold0 modelや旧non-OOF 27列の予測利用
- downstream outer foldを無視したglobal scorer OOF cache
- full-train/batch median、full-train scorer refit
- 依存GRWR 6列、hard top1、weighted TVT、direct blend
- 27列subset、candidate/objective/model/threshold/grid、sample weight
- formation / compact / controlの変更・再学習
- same-OOF救済、gate緩和、無承認package/run/inference/submission

## 実行入口

明示承認後、学習候補を正規train notebookへ採用してpreflightを実行した。
inferenceはfail-closed候補だけで、正規inference notebookは未採用・未実行である。

- 正規学習notebook: `exp396_fold_safe_exp111_score_27_addonly_on_exp287_train.ipynb`
- 推論notebook scaffold: `exp396_fold_safe_exp111_score_27_addonly_on_exp287_inference.ipynb`
- 学習Jupytext候補: `exp396_fold_safe_exp111_score_27_addonly_on_exp287_compact_selfcontained_train.py`
- 学習notebook候補: `exp396_fold_safe_exp111_score_27_addonly_on_exp287_compact_selfcontained_train.ipynb`
- fail-closed推論候補: `exp396_fold_safe_exp111_score_27_addonly_on_exp287_compact_selfcontained_inference.py`
- fail-closed推論notebook候補: `exp396_fold_safe_exp111_score_27_addonly_on_exp287_compact_selfcontained_inference.ipynb`
- 設計: `.steering/20260725-exp396-fold-safe-exp111-score-27-addonly-on-exp287/`

Kaggle Notebook実行を正とする。Stage A kernelは
[`kentookumura/exp396-foldsafe-exp111-score27-exp287-train`](https://www.kaggle.com/code/kentookumura/exp396-foldsafe-exp111-score27-exp287-train)
version 2、id_no `128540844`。Stage B kernelは
[`kentookumura/exp396-score27-exp287-stageb-train`](https://www.kaggle.com/code/kentookumura/exp396-score27-exp287-stageb-train)
version 1、id_no `128570498`。

## 次

exp287をtrain-side parent anchorに維持し、exp396のsubset/grid、same-OOF rescue、
gate緩和、再学習、inference、submissionへ進まない。保存済み生成物だけを使う0-boosterの
転移失敗原因readoutは低・P4とし、新しい独立した必要性と承認がない限り着手しない。

## 表記

用語は `KAGGLE_DIRECTION.md` の表記方針と `docs/glossary.md` に合わせ、実験名や設定名を除いて
日本語優先で記録する。
