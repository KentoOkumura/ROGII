# 要件

## 依頼

exp490をstandalone予測やhard selector候補として採用するのではなく、exp357からexp490へ
変化したgeometry-centered mean-reversion補正だけを、現在の最終anchorへ小さく加える
fold-safeな最終アンサンブル実験を設計する。

初回依頼ではbacklog、steering、実験ディレクトリ、機械可読な設計契約だけを作成した。
2026-08-04の別依頼でStage A実装が承認され、その後の「実行してください」で正規train Notebook採用、
Kaggle package、private CPU Stage A runが承認された。推論と提出は承認範囲外のままとする。

## 仮説

exp490の価値は絶対予測全体ではなく、exp357のpersistent whole-well biasを戻した
`exp490 - exp357`にある。この補正ベクトルを現在のanchorへ10%以下のfold-safeな
非負係数で加えると、exp490の弱い絶対予測やhard routingを持ち込まず、exp413系の
残差と相補的な成分だけを利用できる。

この補正が5 folds、固定scope、by-well tailで安定しなければ、exp490の大きな親比改善は
現在の最終anchorへ一般化する独立成分ではないと判断する。

## 制約

- Routeは`ensemble`とする。最終予測にML anchorとPF/Beam系exact HMM補正が本質的に寄与するためである。
- root parentは`exp413_scale5_likpf_full_replacement_on_exp335`とする。
- 実装開始前に`exp497_strict_public_core_fold_safe_ensemble_on_exp413`のStage Eを終端判定する。
- anchorはexp497の事前登録gateだけで決める。exp497が全AND gateをPASSしてexp413以外をselected predictionにした場合はその保存OOF、その他は保存exp413 OOFを使う。
- exp506のtruth、error、fold score、scope score、by-well scoreを見てanchorを選び直してはならない。
- primary correctionは`d = exp490_prediction - exp357_parent_prediction`に固定する。
- primary predictionは`p = p_anchor + lambda * d`とし、interceptを持たせない。
- `lambda`は他4 outer foldsだけで二乗誤差最小化し、`[0.00, 0.10]`へclipしてheld-out foldへ適用する。
- deployment用`lambda`は5 meta-fold係数の中央値とする。full-OOF再fitは行わない。
- `p = (1-w) * p_anchor + w * p_exp490`はreport-only controlとし、昇格候補や救済候補にしない。
- tau fade、alpha、cutoff、depth bucket、well/row gate、hard router、conditional router、intercept、負weight、3-way stackを使わない。
- exp413、exp497、exp357、exp490、selector、HMM、PF、Beam、ML modelを再学習・再実行しない。
- primary Stage Aは保存OOFだけを使う0-model / 0-booster監査とする。
- Public LB、public test well ID、public row count、submission値、LBに基づくweight選択を禁止する。
- exp490 sourceのtruth/error/episode/gate/by-well outcome列は、primary predictionとSHAをfreezeする前に読まない。
- 実装、Kaggle実行、推論実装、提出はそれぞれ別のユーザー承認を必要とする。
- 再現性は`docs/06_reproducibility.md`に従い、入力・fold・prediction・weight・metricsのSHAを記録する。

## 受け入れ基準

設計完了条件:

- steering 3文書、実験ディレクトリ、`config.yaml`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`が作成されている。
- anchor resolution、入力SHA、join key、primary式、closed-form係数、meta-fold分離、controlの役割が一意に定義されている。
- 実装・実行・推論・提出が未承認であることが明記されている。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`にdesign-only候補として登録されている。

将来のStage A primary all-AND gate:

- technical / leakage / SHA checksが全PASSする。
- cross-fitted pooled RMSE gainがselected anchor比`>= 0.03 ft`。
- fold別RMSEがselected anchor比`5 / 5` foldsで非悪化する。
- `MD 0--250`、`MD 250--1000`、`MD 1000+`、hidden-like spatial、hidden-like typewell-purgedの全5 scopeで非悪化する。
- by-well delta RMSEのp95とworstがselected anchor比でそれぞれ`<= +0.25 ft`。
- 5 meta-foldの`lambda`が全て`0 < lambda < 0.10`で、上限clipへ張り付かない。
- 5 meta-foldの`lambda` rangeが`<= 0.05`。

1条件でもFAILした場合は`FAIL_CLOSE_WITHOUT_WEIGHT_SCOPE_COMPONENT_OR_GATE_RESCUE`とし、
同じOOF上のweight上限、式、component、fade、bucket、router、gateを変更せず、推論・提出へ進まない。

## 終端結果

exp497 Stage Eはgate FAILで終端し、事前ルールによりexp413保存OOFをanchorへfreezeした。
exp506 Stage A version 2はprimary CV`7.902068462119896`、anchor比`+0.017265667715181 ft`悪化、
nonworse`3/5 folds`、固定scope`0/5`、deployment lambda`0.0`となり、事前all-AND gateをFAILした。
`FAIL_CLOSE_WITHOUT_WEIGHT_SCOPE_COMPONENT_OR_GATE_RESCUE`として終端し、推論・提出へ進まない。
