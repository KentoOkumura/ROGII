# 要件

## 依頼

exp238 の selector 学習で使った 184 context 特徴を削除せず、raw test / hidden test
でも同じ定義の特徴を生成する。特に、従来の推論で全行 NaN になっていた
exp109 / exp114 由来の `copcf_*` 41特徴と、exp226 診断4特徴を current test から
再生成し、学習済み outer 5 × inner 4 の20 selectorへ適用できるようにする。

exp245 の41特徴削除版は ablation 履歴として残し、正しい inference port は
exp238 内で管理する。

## 制約

- Route: `ml_model`（exp238 final LightGBM の selector confidence 入力を直す）
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- train schema、保存済み20 selector、outer/inner fold対応、候補定義は変更しない。
- selector 学習、final LightGBM 学習、control再学習は実行しない。
- test well の prior は full train well だけを source とし、test well同士を近傍に使わない。
- test target / hidden label / oracle errorを特徴生成に使わない。
- competition submissionは生成・実行しない。最初は parity audit の5 score面だけを保存する。
- visible testの3 well数に依存しない、well単位で独立な変換にする。

## 受け入れ基準

- selector contextはtrainと同じ184列である。
- `copcf_*` 41列とexp226診断4列が列欠損なくcurrent testに存在する。
- missing context columnは0、exp226診断4列は完全finiteである。
- `copcf_*`は41列すべてを生成し、valid/count/gate等を含む少なくとも32列にfinite値がある。
- trainでも生じ得るprior値/std/cluster距離の自然な部分欠損またはvisible test全体でのcoverage 0は件数を記録し、LightGBM native missing routingへ渡す。
- 保存済み20 selectorのSHA/schemaを検証し、outerごとの4 model平均から5 score面を作る。
- selector fit 0、final booster fit 0、submissionなしをsummaryへ記録する。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## Phase 2: hidden-safe final inferenceへの採用

parity audit通過後は、同じcurrent-test generatorをexp238の保存済みfinal LightGBM推論へ
接続する。既存の正規inference notebookは上書きせず、別名のJupytext notebookで検証する。

- 保存済みselector 20本と保存済みfinal LightGBM 15本を使い、学習は一切行わない。
- selectorはouterごとのinner 4本平均から35 rank-slot特徴を作り、同じouterのfinal 3本へ渡す。
- final feature schemaはexp218 380 + selector 35 = 415列を固定する。
- selector contextは184列、missing列0を必須とし、旧45列NaN fallbackへ戻さない。
- test-test edge/neighborとpublic-test行artifactを使用しない。
- `submission.csv`はKaggle notebook outputとして生成するが、competition submit APIは呼ばない。
- parity context/schema/model/score、final feature schema/prediction/submissionのSHAを保存する。
- このphaseはNaN修正のinference portであり、add-onlyモデル構造自体は変更しない。
