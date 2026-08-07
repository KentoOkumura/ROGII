# 要件

## 依頼

公開ノートブックにあるtrajectory-level後処理を`exp413_scale5_likpf_full_replacement_on_exp335`
へ適用できるか、保存済みexp413 Stage D OOFだけで監査する実験を設計する。

当初はbacklog、steering、実験ディレクトリ、機械可読な設計契約だけを作成した。その後の
ユーザー承認によりStage A実装、正規train Notebook採用、Kaggle package / CPU runまで実施した。
推論、提出は行わない。

## 仮説

exp413の最終TVT予測にはwell内の高周波なtrajectory揺れが残っており、公開実装と同じ
Savitzky--Golay後処理`window_length=61 / polyorder=3`を最終TVTへ一度だけ適用すると、
exp413の物理・ML構成を変えずに小さいがfold横断で安定したRMSE改善を得られる。

公開実装の`tau=85` warmupはexp413では有効性が未確認であり、最終TVTのscale5成分を
再度抑える可能性がある。そのためwarmup単独とwarmup後SGはreport-onlyとし、採否へ使わない。

## 制約

- Routeは`ml_model`とする。exp413の固定ML予測に決定的後処理を加えるだけで、新しいPF/Beam予測を本質的に混ぜない。
- 親は`exp413_scale5_likpf_full_replacement_on_exp335`に固定する。
- controlは保存済みexp413 Stage D OOFとし、exp413の75 boosters、selector、PF/HMM/Beamを再学習・再実行しない。
- selectable primaryは`sg61_p3_final_tvt`の1本だけとする。
- SGは公開sourceと同じくwellごとの保存行順に最終TVTへ適用する。窓は`min(61, n_rows)`、偶数なら1を引き、`window_length >= polyorder + 2`のwellだけSciPy既定modeで処理する。
- SG後のlast-known reanchor、clip、monotonic projection、U-space projection、residual再加算を行わない。
- `tau85_warmup_final_delta`と`tau85_warmup_then_sg61_p3`はreport-only、`selectable=false`、primary救済不可とする。
- 公開の`0.60 * warmup(model) + 0.40 * likPF`は使わない。exp413はすでにscale5 LikPFを全面置換しており、direct LikPFを40%再混合すると仮説がtrajectory後処理監査でなくなるためである。
- SG window/polyorder、tau、blend、clip、reanchor、bucket、row/well gateを同じOOFで探索しない。
- well-level routingはexp508の範囲外とする。公開固定threshold/variant map、test well ID、row count、Public LBを使わない。
- well-level routingはexp508 primaryが全AND gateをPASSし、かつraw/SGのtarget-free disagreementに独立な相補性が確認できた場合だけ、別exp・別steering・別承認で検討する。その場合もthreshold/mapはouter-train wellsだけで再fitする。
- truth、error、scope score、by-well scoreを読む前に入力、row order、3候補predictionをfreezeしてSHAを記録する。
- 実装、Kaggle CPU実行、推論実装、提出はそれぞれ別のユーザー承認を必要とする。
- 再現性は`docs/06_reproducibility.md`に従い、入力・fold・row order・prediction・metrics・decisionのSHAを記録する。

## 受け入れ基準

設計完了条件:

- steering 3文書、実験ディレクトリ、`config.yaml`、`postprocess_contract.yaml`、`output_contract.md`、`README.md`、`SESSION_NOTES.md`、`result.md`、`metrics.json`が作成されている。
- primary、report-only、公開source、入力SHA、row order、truth-late分離、評価scope、gate、禁止事項が一意に定義されている。
- 学習量が0 model / 0 booster / 0 PF/HMM/Beam / 0 GPUであると明記されている。
- 実装・実行・推論・提出が未承認であることが明記されている。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`にdesign-only候補として登録されている。

将来のStage A primary all-AND gate:

- technical / leakage / SHA checksが全PASSする。
- `sg61_p3_final_tvt`のpooled RMSE gainが保存exp413比`>= 0.01 ft`。
- fold別RMSEが保存exp413比`4 / 5` folds以上で非悪化する。
- `MD 0--250`、`MD 250--1000`、`MD 1000+`、hidden-like spatial、hidden-like typewell-purgedの全5 scopeでdelta RMSEが`<= +0.02 ft`。
- by-well delta RMSEのp95とworstが保存exp413比でそれぞれ`<= +0.25 ft`。
- 各well最初のscore rowにおける`abs(SG - exp413)`のp95が`<= 0.50 ft`、最大値が`<= 2.00 ft`。

`0.01 ft`は通常の新規ML候補より小さいgateである。公開source自身がSG効果を約`0.01 ft`と
記述し、今回はparameter探索もmodel fitもない単一決定的変換であるため、この最小改善量を採用する。
代わりにfold、固定scope、well-tail、prediction-start continuityを全ANDで要求する。

1条件でもFAILした場合は`FAIL_CLOSE_WITHOUT_SG_GRID_WARMUP_ROUTER_OR_GATE_RESCUE`とし、
同じOOF上でwindow/polyorder/tau/reanchor/clip/router/gateを変更せず、推論・提出へ進まない。

## 次のアクション

2026-08-04の追加依頼で正規Notebook採用とKaggle private CPU Stage Aまで完了した。
SG61/p3は`0.006133728 ft`改善したが固定`0.01 ft` gateを未達としたため、branchを終端閉鎖する。
推論、提出、router、same-OOF救済は行わない。
