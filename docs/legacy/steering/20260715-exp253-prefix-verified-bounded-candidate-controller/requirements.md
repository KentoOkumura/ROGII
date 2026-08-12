# 要件

## 仮説

visible prefix上で既存candidateを評価し、十分に信頼できるwellだけ補正量を制限してbaseを動かすと、
exp238のworst-well回帰を抑えながらoverall RMSEを改善できる。

## 依頼

公開 notebook `yusuketogashi/rogii-another-approach` から、visible prefix で既存候補を評価し、
信頼できる候補方向へ補正量を制限して予測を動かす controller だけを導入する。
Beam/PF 本体、候補生成アルゴリズム、contact reconstruction、heel calibration、GR alignment、
bimodal correction は変更しない。

## 制約

- Route: `ensemble`
- 再現性: `docs/06_reproducibility.md` に従い、stochastic 処理、PF/Beam、GPU 学習、Kaggle bootstrap、SHA 記録の扱いを設計に明記する。
- 親の保存済み OOF / candidate cache を control とし、親モデルを再学習しない。
- masked-prefix replay は既存 exp072 PF/Beam/likelihood-PF 実装を再利用し、新しい candidate family を作らない。
- prefix cut は `0.50 / 0.65 / 0.75`、balanced controller は `alpha <= 0.40`、row move は `<= 30 ft` に固定する。
- same-well contact、formation contact、train-only columns、evaluation-tail true TVT による candidate 選択を禁止する。
- 32 wells の Stage 0完了後、ユーザー判断によりworst-well回帰は拒否条件から監視指標へ変更する。
- 773-well Stage 1は科学設定を変えず、stable SHA256 well modulo 4のCPU shardで実行する。
- shard単独では採用・推論を許可せず、4 shardのrow-level OOFを結合してglobal RMSEを再計算する。

## 受け入れ基準

- Stage 0 / Stage 1 の双方を実行できる Jupytext train notebook と、train guard 不通過時に停止する inference notebook が実装されている。
- masked prefix より後の `TVT` は candidate generation に渡らず、prefix holdout scoring と official-tail evaluation にだけ使われる。
- candidate score、gain、rank margin、cut consistency、alpha、clip、適用行数を well 単位で保存する。
- control と controller の overall / distance bucket / hidden-like / fold / worst-well 指標を保存し、全 safety guard 通過時だけ inference を許可する。
- worst-wellは値と旧0.25 ft判定を保存するが、2026-07-15のユーザー判断により最終passの必須条件には含めない。
- 4 shardのwell集合が重複せず全773 wellsを覆い、入力/config SHAが一致する場合だけaggregateを成立させる。
- variant 1、LightGBM config 0、fold training 0、booster 0、parent/control 再学習なしが config と notebook 上で確認できる。
- deterministic anchor として扱う場合は、feature content SHA、model SHA、prediction SHA、submission SHA、Kaggle kernel version が記録されている。
- gzip 生成物を比較する場合は、raw `.csv.gz` SHA ではなく decompressed content SHA を主証拠として記録している。

## 次のアクション

4つのStage 1 CPU shardを実行し、完了後にaggregate notebookで全well指標と採用guardを判定する。
