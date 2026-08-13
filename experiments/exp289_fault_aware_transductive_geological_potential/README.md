# exp289_fault_aware_transductive_geological_potential

## 状態

- ルート: `pf_beam`
- 状態: Stage 0科学guard不通過・branch closed
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-19
- 親実験: なし。新規standalone physics family

## 仮説

formation surfaceはwithin-wellで`F = Z + TVT + c_w`を満たすため、未知suffixのTVTは
共通2D surfaceのanchor-relative変位から一意に求められる。exp226の損失を支配する少数wellの
large biasが断層を跨いだ空間平均に由来するなら、fault cutを許したpiecewise-smooth surfaceを
全hidden-like well同時にMAP推定することで、候補bankやoracleなしにtailを減らせる。

## モデル境界

- canonical surface: outer-train `ANCC`
- hidden-like observation: known prefixの`Z + TVT_input`とlatent well datum
- model: fault-aware truncated-quadratic graph + deterministic IRLS
- output: 一つのdirect MAP TVT prediction
- 使用しないもの: ML、既存予測入力、blend、candidate selector、row/well oracle、posthoc補正
- source ANCC欠損: 全行非有限wellだけdonorから除外し、部分欠損はfail-closed。除外内容をfold別manifestへ記録する。
- Stage 1: GRなし
- Stage 2: Stage 1 guard通過・別承認後だけknown-prefix calibrated ordered GR event factorを追加

## 段階とguard

1. Stage 0: target-free fault-riskをfreezeし、`abs(exp226 bias)>=10` AUC 0.65、Spearman 0.25、4/5 folds正方向を要求する。
2. Stage 1事前guard: direct OOF 8.0以下、4/5 folds改善、well p95 15以下、固定worst-66 MSE share 45%以下。
3. 単独inference guard: direct OOF 7.0以下、5/5 folds改善、hidden-like 2面改善、well p95 13以下、worst 40以下。

どの段階も不通過時はparameter救済grid、GR追加、inference、submissionへ進まない。

## 検証方針

- Fold: 既存5-fold GroupKFoldを固定
- Group: well単位
- Transductive scope: fold内outer-valid wellsを全件同時推定
- Score rows: `TVT_input.isna()`
- Leakage check: outer-valid formation 6列とprediction-target true TVTをgraph/scale/fault/solver fit前にdropし、truthはcontent SHA freeze後にだけjoin
- 比較: 保存済みexp226 OOF。control再生成なし
- oracle: 全粒度で禁止

## 実行入口

- 学習 notebook: `exp289_fault_aware_transductive_geological_potential_train.ipynb`（Stage 0 compact self-contained実装）
- 推論 notebook: `exp289_fault_aware_transductive_geological_potential_inference.ipynb`（Stage 1承認前はfail-closed）
- 編集元: `*_compact_selfcontained_train.py` / `*_compact_selfcontained_inference.py`
- Kaggle実行: private CPU version 3完了、runtime 241.548秒、peak RSS 693.191 MB
- canonical kernel: `kentookumura/exp289-fault-aware-geopotential-stage0-train`

## 結果

| メトリック | 値 |
| --- | --- |
| `abs(exp226 bias)>=10` AUC | 0.570652（guard 0.65未達） |
| pooled Spearman | 0.127885（guard 0.25未達） |
| 正方向fold | 5/5（guard 4/5を通過） |
| technical guard | 全項目PASS |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- 物理式、単一MAP出力、transductive CV、fault仮説の反証guardを実装前に固定した。
- oracle、候補bank、selector、blendを明示的に実験範囲外へ置いた。
- outer-validは`MD/X/Y/Z/TVT_input`だけを読むsafe loaderとし、fault riskをSHA freezeした後だけexp226 errorとformation 6面を読む順序をコードとtestsで固定した。
- cross-well kNN edgeをdistance / source well / source rowでstable化し、fold別node/edge/risk content SHAを出す。
- 773 wells / 320,991 node risksを生成し、outer-valid forbidden column hit 0、truth-before-freeze 0、finite coverage 100%を確認した。
- fault riskは5/5 foldsでexp226 biasと正方向だった。

### 悪かった点

- AUC 0.570652とSpearman 0.127885は事前guardに届かず、whole-well large biasの識別力が不足した。
- Stage 1 sparse MAP solverはguard不通過のため未実装のまま閉じた。

## リスク / 注意

- train-only formationをouter-validへ残すと重大なleakageになる。
- fault仮説が誤りならStage 0で閉じる。
- ordinary smooth KNN/GPへ戻す救済は行わない。
- 同じOOFでedge threshold、formation面、risk aggregationを探索する救済は禁止する。
- deterministic anchorではない。v3 source/configと生成物SHAは照合済みだがrerun parityは未確認。

## 次

事前failure policyに従いexp289 branchを閉じる。Stage 1/2、inference、submissionへ進まず、同一fault-riskのparameter救済も行わない。

## 表記

用語は`backlog/KAGGLE_DIRECTION.md`の表記方針と`docs/glossary.md`に合わせ、実験名や設定名を除いて日本語優先で記録する。
