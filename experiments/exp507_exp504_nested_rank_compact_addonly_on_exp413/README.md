# exp507_exp504_nested_rank_compact_addonly_on_exp413

## 状態

- Route: `ensemble`
- 状態: Stage N technical PASS・Stage D technical PASS / scientific FAIL・終端閉鎖
- 親 / matched control: `exp413_scale5_likpf_full_replacement_on_exp335`
- rank source: `exp504_h512_regret_weighted_block_rank_selector`
- control CV: `7.884802794404715`
- exp507 CV / Public LB / Private LB: `7.889515565580203` / 未提出 / 未提出
- 作成日: 2026-08-03

## 仮説

exp504はhard winnerへ変換した後のtail安定性に失敗したが、pairwise勝敗面まで無価値とは限らない。
H512 blockごとのBorda、anchor勝率、分布要約、候補TVTのweighted momentをcompact特徴として
exp413の後段TVTモデルへ渡すと、hard choiceを強制せずrank情報を利用できる可能性がある。

## 確定した変更

- exp413の`clean273 + compact74 + signed23 = final370`はそのまま残す。
- exp504 rank面を45列へ圧縮し、add-onlyの`final415`を1 treatmentだけ評価する。
- 66 pair全部は渡さず、anchorとの11勝率だけを渡す。
- provisional候補はcandidate ID 1列ではなく12列one-hotにする。
- anchor scoreは12 Borda列内のanchor列を使い、重複列を作らない。
- Borda-weighted TVT平均・標準偏差は特徴としてだけ使い、直接予測にしない。
- H512内相対位置をrowごとに1列追加する。

## nested契約

exp413自体はnestedである。一方、exp504の保存OOFはstandard outer 5-foldなので、それを
exp413のouter-train全行へ貼るとinner cross-fitにならない。将来のStage Nでは、各downstream
outer foldについてouter-validは保存exp504出力を再利用し、outer-trainはheld outerとheld innerを
除く3 foldsで20本のrank modelを学習して25 compact partitionsを作る。

## 検証方針

- Fold / Group: exp413と同じwell-grouped outer 5 folds。
- rank compact: outer-validは保存exp504 surface、outer-trainはouter 5 × inner 4のstrict cross-fit。
- matched control: 保存exp413 OOF `7.884802794404715`。control boosterは再学習しない。
- primary gate: gain `>= 0.03 ft`、非劣化`>= 3/5 folds`、固定5 scopeが
  exp413比`<= +0.02 ft`、technical/leakage/SHA全PASS。
- by-well p95/worstと`+1/+3/+5 ft`悪化well数は必須readoutとして残す。

## 実装

- 別名Jupytext source / Notebook:
  `exp507_exp504_nested_rank_compact_addonly_on_exp413_compact_selfcontained_train.py/.ipynb`
- Stage N: exp504 frozen target-free surfaceのSHA preflight、20 strict nested CPU rank model、
  25 compact partition、technical/leakage/model/partition manifestを実装済み。
- Stage D: Stage N manifest SHAを必須入力とし、exp413 final370へrank45をadd-onlyして
  final415を3 configs × 5 foldsで学習・固定gate判定する処理を実装済み。
- contract test: 7件PASS。Jupytext round-trip、py_compile、Ruff、strict validationもPASS。
- `block-constant 44 / row-varying 1`という設計上の数え違いは、row単位weighted moment式を
  正として`42 / 3`へ訂正した。45列の名前・順序・式は変更していない。

## 所見

Stage D final415は保存exp413より`0.004712771 ft`悪化し、fold非悪化`2/5`、最大scope delta
`+0.036938807 ft`で固定性能gateをすべてFAILした。pair surfaceを45列に圧縮して後段へ渡しても、
exp413を安定して改善できなかった。

## 実行量

- Stage N: 1 rank config × outer 5 × inner 4 = 20 CPU boosters
- Stage D: 1 treatment × TVT LightGBM 3 configs × outer 5 = 15 GPU boosters
- 合計35 boosters
- exp413 control、exp504 outer model、候補/PF/HMM/Beamの再学習・再生成: 0

## 利用可否

Stage N rank compact生成物とStage D final415はtechnical PASSしたが、Stage Dはscientific FAIL。
`FAIL_CLOSE_WITHOUT_PAIR_FEATURE_SUBSET_TEMPERATURE_OR_GATE_RESCUE`として利用不可・終端閉鎖し、
inference / submissionは実装・実行しない。
詳細な45列schema、nested split、promotion gate、禁止事項はsteeringの`design.md`を正とする。

## 実行入口

2026-08-03のStage N実行指示を正規train Notebook採用承認として記録し、別名のJupytext
compact self-contained候補を正規train Notebookへ反映する。inference notebookは
markdown-only placeholderのまま保持する。

## 次

CPU Stage N version 1（id_no `129565024`）とT4 Stage D version 1（id_no `129584313`）は完了。
Stage Dは15 models、control再学習0でtechnical PASSしたがscientific FAILのため、これ以上進めない。
