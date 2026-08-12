# 要件

## 依頼

`exp504_h512_regret_weighted_block_rank_selector` のhard-selected TVTをそのまま採用せず、
pairwise rank modelが持つH512 block内の相対評価をcompactな連続・カテゴリ特徴へ変換し、
`exp413_scale5_likpf_full_replacement_on_exp335` の後段TVT LightGBMへadd-onlyで渡す
train-side実験を設計する。

今回はbacklog、steering、実験ディレクトリ、機械可読な設計契約だけを作る。実装コード、
Jupytext source、Notebook実装、Kaggle package、学習、推論、提出は行わない。

2026-08-03の追加依頼「exp507を実装してください」により、train-sideのJupytext compact
self-contained候補、Stage N / Stage Dコード、専用contract test、別名Notebookまでを追加承認した。
Kaggle package、Stage N、Stage D、推論、提出の実行承認は含まない。

2026-08-03の追加依頼「実行してください」により、直前に次工程として提示したStage Nの
正規train Notebook採用、Kaggle CPU package、push、20 CPU booster runを承認した。
Stage Dの15 GPU boosters、推論、提出は承認対象外のままとする。

2026-08-03の追加依頼「次に進んでください」により、Stage N technical PASS後の次工程として
提示したStage DのKaggle T4 package、push、1 treatment × 3 configs × 5 folds = 15 GPU booster
runを承認した。保存exp413 control、Stage N、候補/PF/HMM/Beamは再学習・再生成せず、
inferenceとsubmissionは承認対象外のままとする。

## 仮説

exp504のhard selectorはpooled RMSEをanchor比`0.124054566 ft`改善した一方、fold、
hidden-like、by-well tailで失敗した。しかしweighted pair accuracyは`0.741908`であり、
勝敗面にはhard top-1へ潰す前の情報が残っている可能性がある。66 pair確率をそのまま渡さず、
Borda面、anchorとの比較、順位の集中度、候補値のweighted moment、block内位置に圧縮すれば、
後段TVTモデルが状況に応じてrank情報を弱く利用し、exp413より安定して改善できる可能性がある。

## 制約

- Routeは`ensemble`とする。exp413のML予測と、exp504の物理候補bankに基づくrank面が
  最終TVT予測へ本質的に寄与する実験だからである。
- root parent / matched controlは`exp413_scale5_likpf_full_replacement_on_exp335`とする。
- rank sourceは完了済みexp504 Kaggle CPU version 1の科学条件に固定する。
- exp413の`clean273 + compact74 + signed23 = final370`を削除・置換せず、rank compact 45列を
  add-onlyで追加して`final415`とする。
- downstreamへ66本の全pair確率を渡さない。anchorとの11 pairだけを渡す。
- anchorは`exp226_w500_50_50`、candidate順、H512 block、pair loss/weight、LightGBM config、
  Borda、tie rule、0.5 anchor guardをexp504から変更しない。
- anchor scoreは12本のBorda列のうちanchor列そのものとし、重複列を追加しない。
- provisional candidateはordinal ID 1列で表さず、固定candidate順のone-hot 12列とする。
- Borda weightは`w_j = borda_j / sum_k(borda_k)`に固定し、temperatureやsoftmaxを使わない。
- outer-validのrank面はexp504 version 1の保存OOF artifactをSHA固定して再利用し、
  完了済み5 outer rank modelを再学習しない。
- downstream outer-train行は、held downstream outer foldとheld inner foldの両方を除いた
  3 foldsだけで学習するinner rank modelによりcross-fitする。standard 5-fold OOFを
  outer-trainへ複製してはならない。
- 将来のrank nested生成は1 rank config × outer 5 × inner 4 = 20 CPU boosters、
  downstreamは1 treatment × LightGBM 3 configs × outer 5 = 15 GPU boostersに固定する。
- exp413 control 15 boosters、exp504 outer 5 boosters、候補予測、PF/HMM/Beamは再学習・再生成しない。
- hard-selected TVT、selected candidate ID、66 pair全部、truth/error/oracle、well IDを特徴にしない。
- feature subset、pair subset、temperature、threshold、loss、weight、rank model、TVT model config、
  candidate、block horizonのgridを行わない。
- 実装、rank nested生成、downstream学習、推論、提出はそれぞれ別のユーザー承認を必要とする。
- 再現性は`docs/06_reproducibility.md`に従い、入力、schema、partition、model、predictionの
  SHAを記録する。

## 受け入れ基準

設計完了条件:

- steering 3文書、実験ディレクトリ、`config.yaml`、`README.md`、`SESSION_NOTES.md`、
  `result.md`、`metrics.json`が作成されている。
- 45列の名前、順序、数式、dtype、blockからrowへのbroadcast、nested splitが一意である。
- exp413がnestedであることと、exp504 standard OOFをそのまま追加するとexp413のouter-train側で
  leakageになることの違いが明記されている。
- 将来の実行量が20 CPU rank boosters + 15 GPU TVT boosters、control再学習0として固定されている。
- 実装・実行・推論・提出が未承認であることが明記されている。
- `KAGGLE_DIRECTION.md`と`experiment_summary.md`にdesign-only候補として登録されている。

実装完了条件:

- exp504 v1の必要生成物についてfile / logical SHAが固定され、実ファイルpreflightが通る。
- strict nested 20 model / 25 partition plan、compact45 builder、Stage N technical gate、
  final415 Stage Dがコード化されている。
- Jupytext round-trip、py_compile、Ruff、専用test、strict experiment validationがPASSする。
- 正規train Notebookは上書きせず、別名compact self-contained候補として保持する。
- Stage N / Stage D run flagはfalse、Kaggle package / inference / submissionは未作成のままとする。

Stage N実行承認後の次アクションは、20 CPU rank boosters、control再学習0を再確認し、
canonical CPU kernelをpackage/pushしてtechnical gateを判定すること。

将来のStage N technical gate:

- rows / wells / outer folds / H512 blocksが`3,783,989 / 773 / 5 / 7,787`と一致する。
- outer-valid 5 partitionsはexp504 version 1の保存artifactと候補順・Borda・anchor pairが一致する。
- outer 5 × inner 4の20 modelと、4 inner-train + 1 outer-validからなる25 compact partitionsが揃う。
- 25 partitionsは合計`18,919,945` row-role、45列、重複名0、NaN/Inf 0、key欠損0である。
- 各inner partitionのrank model学習wellにheld outer / held inner wellが0件である。
- ordinal candidate ID、hard-selected TVT、66 pair全列、target/error/oracle列が0件である。
- schema、content、partition、model manifest、全20 model SHAが保存される。

将来のStage D primary all-AND gate:

- technical / leakage / SHA checksが全PASSする。
- pooled OOF RMSEが保存exp413 `7.884802794404715`から`>= 0.03 ft`改善する。
- fold別RMSEがexp413比`>= 3 / 5` foldsで非悪化する。
- `md_since 0--250`、`250--1000`、`1000+`、hidden-like spatial、
  hidden-like typewell-purgedの全5 scopeがexp413比`<= +0.02 ft`である。
- by-well delta p95 / worstと`+1/+3/+5 ft`悪化well数を必須readoutとして保存する。

1条件でもFAILした場合は`FAIL_CLOSE_WITHOUT_PAIR_FEATURE_SUBSET_TEMPERATURE_OR_GATE_RESCUE`
とし、same-OOF上のcompact列subset、pair追加、weight、threshold、model、gateを変更せず、
推論・提出へ進まない。

## Stage D確定結果（2026-08-03）

private T4 version 1（id_no `129584313`）で15 GPU boostersを完走した。technical checksは
PASSしたが、exp507 / exp413 pooled RMSEは`7.889515566 / 7.884802794`、gain
`-0.004712771 ft`、nonworse folds `2/5`、最大scope delta `+0.036938807 ft`で、固定した
性能3条件をすべてFAILした。したがって上記fail actionを適用し、same-OOF rescue、
inference、submissionなしで終端閉鎖する。
