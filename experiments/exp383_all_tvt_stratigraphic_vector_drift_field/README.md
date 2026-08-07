# exp383_all_tvt_stratigraphic_vector_drift_field

## 状態

- ルート: `pf_beam`
- 状態: Stage 0 code/resource FAIL・full run停止
- CV: -
- Public LB: -
- Private LB: -
- Submit ID: -
- 作成日: 2026-07-24
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`

## 仮説

outer-train坑井の正解TVT全体から`S=TVT+Z`の絶対場と2次元水平勾配場を作り、
6つのtrain-only地層面で条件付ければ、exp226の疎なK16 scalar donorより
対象坑井の長い未知suffixへ地層ドリフトを移送できる。

## 変更点

- 提供PS以降のK16 12,368区間ではなく、全TVTを覆う64/256/1024 ft windowを教師にする。
- スカラー傾きコピーを、坑井方位へ射影できる2次元vector fieldへ変える。
- 6地層面の値、dip/strike、厚さ、補間分散をfold-safeに使う。
- 全既知prefixの`TVT_input+Z`で単一vertical biasを校正する。
- support/uncertaintyに応じてexp226 rateへ連続縮約する。
- absolute/vector/known-prefixを制約付きbanded solveで1本の物理pathへ戻す。

## 検証方針

- Fold: exp226 outer 5-fold
- Group: well
- Control: 保存済みexp226 OOF、CV `9.427109596582213`
- Stage 0: target-free surface/catalog/field/support/resource監査
- Stage 1: exp226比`>=1.0 ft`、4/5 folds、1000+/hidden-like改善
- Leakage Check: outer-validの生Formationとsuffix TVT read 0、donor self-exclusion
- 現在見えている3 test wellsは設定選択に使わない。

## 実行入口

- `*_compact_selfcontained_train.py` / `.ipynb`にStage 0/1実装済み。
- trainはouter-validの生Formation/TVTを読まず、target-free SHA freezeとStage 0 PASS後だけ
  exp226 OOF truthをlate joinする。
- `*_compact_selfcontained_inference.py` / `.ipynb`は推論・提出をfail-closedにする。
- ユーザーの実行指示によりcompact候補を正規train/inference Notebookへ採用した。
- version 1はprivate CPU / internet off / run-on-pushで実行した。
- ローカルpackageは修正版、`run_on_push=false`、execution無効へ更新済み。
- canonical kernel: `kentookumura/exp383-tvt-vector-drift-field-train`
  version 1 / id_no `128459031`、status `ERROR`
- 推論、提出は無効のまま。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |

## 所見

### 良かった点

- LB 6.5を狙う物理モデル本命として、exp226の小幅reweightではない構造変更を固定した。

### 悪かった点

- version 1は約6.13時間後、multiscale donorのsurface joinに一意node IDがなく
  `MergeError`で停止した。joinはscale込み`query_id`で修正し回帰testを追加済み。
- fold 0の209,467 donor windowsに対する実測から、5 foldのsurface stageだけで
  約109,867秒（30.52時間）と投影され、固定上限30,600秒の3.59倍である。
- truth join前の失敗なのでCVはなく、Stage 0 resource FAILとしてfull runを停止した。

### リスク / 注意

- surface外挿、方位condition、非保存vector field、CV/LB不一致が主要リスク。
- 初回科学readoutではby-well tailを報告するが、pooled signalを隠す停止条件にはしない。

## 次

- exp383の再push、full run、Stage 1、inference、submissionは行わない。
- donor surfaceの計算方式を根本的に変える場合は、同一OOFでの救済ではなく
  新しい実験契約とresource preflightを別途設計する。
