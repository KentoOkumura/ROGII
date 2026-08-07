# exp334_equal_well_loss_weighting_on_exp287

## 状態

- ルート: MLモデル
- 状態: Kaggle train完了、固定promotion guard不通過、非昇格でclose
- CV: `8.09349752413077`（exp287比`-0.04321069622868201 ft`）
- Public LB: 未実行
- Private LB: 未実行
- Submit ID: なし
- 作成日: 2026-07-21
- 親実験: `exp287_fold_safe_formation_74_addonly_on_exp264`

## 仮説

exp287では、評価行が多いwellほどrow単位L2損失への寄与が大きい。この偏りが、global RMSEを改善しながら一部wellを大きく悪化させた原因の一つなら、各wellの総学習重みを同じにするだけでglobal gainをほぼ維持しつつwell-level tailを改善できる。

この因果説明は未検証の仮定であり、exp334の固定gateで判定する。

## 変更点

- outer-trainのwell `w` の行数を `n_w`、総行数を `N`、well数を `W` とし、各行へ `N / (W * n_w)` のsample weightを付ける。
- 各wellの総重みを `N / W`、全行の平均重みを1にする。
- validにはweightを付けず、early stoppingと全評価指標は非加重RMSEのままにする。

変更は上記1点だけで、exp287の421特徴、fold、target、3 LightGBM config、seed、selector、formation生成は固定する。hard-well errorやinner-OOF errorを使う再重み付けは本実験に含めない。

## 検証方針

- Fold: exp287と同一の5-fold outer group split
- Group: `well`
- 評価対象: `TVT_input_isna`
- 主control: 保存済みexp287 OOF（RMSE `8.136708220359452`）
- tail control: 保存済みcorrected exp264 OOF（RMSE `8.460811237612477`）
- Leakage check: weightはouter-trainのfold、score-row identity、well別行数だけから計算し、target、prediction、error、outer-valid rowsを参照しない。

全promotion gateは[steering設計](../../.steering/20260721-exp334-equal-well-loss-weighting-on-exp287/design.md)を正とする。主要条件は、pooled RMSEがexp287比`+0.02 ft`以内、4/5 folds以上が非悪化、by-well delta p95が非悪化、exp264比worst-wellが`+0.25 ft`以内、`+1/+3/+5 ft`悪化well数が`135/39/14`以下である。

## 実行量と承認状態

- 新規variant: 1
- LightGBM config: 3
- folds: 5
- 実績: 15/15 GPU boosters完了
- control再学習: 0
- 現在: compact正規採用、0-booster preflight、15 GPU boostersを完了。control再学習は0。guard不通過のためtrainを閉じ、inferenceとsubmissionは未実行・未承認。

## 実行入口

次のJupytext percent形式候補を実装し、2026-07-21の実行承認でtrain候補の正規notebook採用が承認された。

- `exp334_equal_well_loss_weighting_on_exp287_compact_selfcontained_train.py` / `.ipynb`
- `exp334_equal_well_loss_weighting_on_exp287_compact_selfcontained_inference.py` / `.ipynb`

train候補は`preflight_only`と承認後の`equal_well_weight_train`を持つ。inference候補はtrain promotion PASSと別承認までfail-closedで、sample submissionも生成しない。まず正規train notebookをpreflight stageでpackageし、PASS後だけtrain stageへ切り替える。

正規train notebookへの採用は完了した。22 cells（code 10 / markdown 12）、保存output 0で、SHA256は`4b5c1a48503422742b9c49cdd315dc85244b51fb5aebc087224f0c77afdc90a5`。

## 結果

Kaggle上の0-booster preflight version 1（id_no `128110184`）は`647.994780625 sec`でPASSした。version 2は15/15 boostersを`21882.805369142 sec`で完了し、CV `8.09349752413077`、exp287比`-0.04321069622868201 ft`、5/5 folds改善だった。全scopeもgate内だったが、by-well p95 `+0.429584617 ft`、exp264比worst-well `+7.156485377 ft`、`+3/+5 ft`悪化well数`40/19`が条件を満たさず、固定AND gateはFAILした。LBは未実行。

## 所見

### 最終判断

well均等lossはglobal、全fold、長距離、hidden-likeを改善し、worst-wellもexp287から約`1.07 ft`軽減した。ただしsevere tailをclean control水準へ戻せず、仮説は部分的支持に留まる。exp334は非昇格として閉じ、weight grid、追加train、inference、submissionは行わない。

### Gate結果

pooled、5/5 folds、全scope、`+1 ft`悪化well数はPASS。by-well p95、worst-well、`+3/+5 ft`悪化well数はFAIL。詳細は`result.md`と`metrics.json`を正とする。

## 成果物

- 設計の正: [steering設計](../../.steering/20260721-exp334-equal-well-loss-weighting-on-exp287/design.md)
- 設定の正: `config.yaml`
- 実行記録: `SESSION_NOTES.md`
- Kaggle train outputにはOOF、15 models、metrics、fold/scope/hidden/by-well metrics、manifestがある。全modelと主要成果物のSHAを監査済み。predictionとsubmissionは生成していない。

## リスク / 注意

- 長いwellの寄与を弱めるため、pooled RMSEが悪化する可能性がある。
- Public testは3 wellsであり、CVのwell均等化効果がPublic LBへ移る保証はない。
- 重みへtarget/errorを混ぜるとOOF overfitになるため禁止する。
- gate不通過時にweight式やguardを同一実験内で調整しない。

## 次

exp334は非昇格で完了。既存バックログの0-booster formation tail attribution readoutは再開条件を満たしたが、着手は別途ユーザー確認後とする。
