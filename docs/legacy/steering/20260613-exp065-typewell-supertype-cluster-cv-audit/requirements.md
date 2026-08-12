# 要件

## 依頼

`typewell_supertype_cluster_cv_audit` を実装する。ただし主目的は CV の補助監査ではなく、train wells 間で共通 typewell と見なせる候補を見つけること。

Discussion 705210 の「PNG では 57 unique Typewell numbers」という主張と、既存の exact CSV duplicate audit で 752 unique CSV が出た事実の間を埋めるため、CSV の typewell GR 曲線から exact duplicate、native row-lag overlap、shifted NCC 類似、constrained DTW 類似のグループを出す。

## 制約

- Route: `pf_beam`
- CV スコアや提出候補の選定を目的にしない。
- クラスタ数を 57 に合わせて最適化しない。57 付近になる閾値があっても、診断結果として記録するだけにする。
- validation well の target や OOF 予測を使わず、`*__typewell.csv` のみから共通 typewell 候補を作る。
- exact duplicate、shifted NCC、DTW は別々に保存し、混同しない。
- native row-lag overlap は、投稿で示唆された shift / trim 一致を検出する主経路として扱う。

## 受け入れ基準

- train typewell CSV を読み、well ごとの exact hash と曲線シグネチャを保存する。
- shifted NCC の上位ペアと閾値別 connected components を保存する。
- NCC 候補ペアに constrained DTW をかけ、DTW 類似ペアと閾値別 connected components を保存する。
- native `typewell.csv` の GR 列を row-lag で照合し、overlap rows、exact match rate、row lag、ft 換算、containment 関係を保存する。
- group ごとに代表 well、members、size、exact duplicate との関係を保存する。
- `metrics.json` に unique group 数、最大 group、57 付近の閾値有無、生成物パスを記録する。
