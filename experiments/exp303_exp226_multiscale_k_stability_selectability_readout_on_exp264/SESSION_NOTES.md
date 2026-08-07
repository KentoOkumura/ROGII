# exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264 セッションノート

## 目的

exp302のK12/K16/K24 prediction-only安定性が、corrected exp264 Stage C v6のK16 misrankingを
fold-stableに識別できるか、固定H512 scoreで監査する設計を確定する。

## 現在の状態

- Route: `ml_model`
- 状態: Kaggle private CPU version 1完了・technical PASS / scientific FAIL・branch閉鎖
- CV/LB: pooled H512 AUC `0.488805`、AUC>0.5は`1/5 folds` / LBなし
- 実装承認: 2026-07-21、ユーザーの`exp303を実装してください`を継続依頼として適用
- Kaggle実行承認: 2026-07-21 13:37:37 JST、ユーザーの`実行してください`を適用
- 承認scope: 正規Notebook採用、1 private CPU readoutのpackage/push/run/監視、結果記録。inference/submissionは対象外
- inference/submission: 対象外
- dependency: exp302 technical/novelty PASS、exp276 corrected-parent完了+promotion guard FAILの全4条件成立

## 2026-07-20 設計確定

- exp300のselection-regret SSE 52.3%というK16 misranking evidenceを背景にした。
- ただしoracle情報はfeatureへ入れず、level/slope/boundaryの3固定成分だけでprimary scoreを定義した。
- primary scoreの方向を「高instabilityほどK16がunderselectedされる」に固定し、truthで反転しない。
- primary unitを非重複H512 block、positive labelをK16の`>=0.25 ft` block RMSE優位に固定した。
- exp302 novelty PASSかつexp276 FAILを開始条件にした。
- 親exp264/exp302のコードやNotebookはコピーしていない。

## 実行量（dependency成立後に実装された場合）

| 項目 | 数 |
| --- | ---: |
| fixed readout variants | 1 |
| evaluation folds | 5 |
| trained folds | 0 |
| LightGBM configs | 0 |
| boosters | 0 |
| candidate regeneration | 0 |
| parent retraining | 0 |
| GPU | 0 |

## 再現性メモ

- seed/RNG: primary readoutに乱数なし。
- CPU/GPU: CPU、初回`num_workers=1`、GPUなし。
- exp264 input: corrected Stage C v6 candidate score SHA
  `a10b7848127f01bef522f4b17dfd1640c9784956892dc24fc1159e3869500abc`を要求。
- exp302 input: K12/K16/K24 prediction content SHAを全一致させる。
  - K12 content/decompressed: `c3d7dfe20ad3b8c7d6d5220023bbb4526fb90d10cc73f01e612db847af70da63` / `63b381299ee46fa172680af57959d675c68b6b24af05664c8689dd291961f22d`
  - K16 decompressed: `709eb726cc30da523f017ed0dbd0371967b88a91ddcf25578eb9356f28e4c609`
  - K24 content/decompressed: `dca92e8f21d3b8b33d1543fe3df0bf586be3a2604b76ee1bf19fa84a327f06ef` / `ca36d168b45acb15cc814ac3c1c3437894cd1050f6c51ba03f5b302efd0a31aa`
- freeze: feature schema/value、outer-train empirical maps、primary score、H512 blockをtruth join前に固定。
- model/prediction/submission SHA: 新規model/prediction/submissionを生成しないため対象外。
- deterministic anchor: false。diagnostic readoutでありprediction anchorではない。

## 2026-07-21 dependency更新

- exp302 Kaggle private CPU version 2はtechnical PASS、K12/K24 candidate novelty PASSとなり、exp302側の2条件を充足した。
- direct guardは両variant FAILだが、exp303開始条件は事前登録どおりcandidate noveltyだけを参照する。
- exp276 corrected-parent Kaggle version 3はtechnical PASS / fixed q70/q80/q90 guard全FAILで完了した。
- 4 dependencyが全成立し、元のexp303実装依頼に基づいて実装を開始した。

## 2026-07-21 実装

- 1,548行 / 18 cellsの`*_compact_selfcontained_train.py/.ipynb`と、fail-closed inferenceを別名で作成した。
- 実装時点では既存の正規train/inference Notebook placeholderを上書きせず、別名版として検証した。
- exp302 freeze manifest SHA `bd80a4e...b6919`を追加し、保存前prediction content宣言と
  K12/K24保存gzip decompressed SHAを別々に検証する。
- K12/K24 3,783,989行をstream読込するpreflightを実施し、保存CSV parsed content SHAは
  K12 `4cb91ea5...3e05f`、K24 `4b9c1473...32b80`だった。これは保存前content SHAの代替ではなく、
  exp303側freeze evidenceとして記録する。
- K12/K24/K16を合わせたtruth-free loaderも実ファイルで完走し、3,783,989 rows / 773 wells、
  3 prediction列finite coverage 1.0、well-row alignment一致を確認した。
- K16はtruth/error列をusecolsへ入れず、well/row/suffix/pred/foldだけを読む。
- corrected Stage C v6 candidate-longもactual error列を読まず、primary 11候補のpredicted errorだけで
  selected hardを復元する。
- level、H128 slope、continuous K boundary±8のjump featureを固定し、各corrected Stage C outer foldの
  train 4 foldsだけをreferenceにmid empirical CDFを作る。
- row scoreは3成分平均、primary H512 scoreはrow p90。truth-free parquet/schema/preprocessor/block/manifestを
  SHA freezeし、再計算最大差`<=1e-12`を必須にした。
- raw TVTとhidden-like assignmentはfreeze検証後だけ読み、pooled/fold/quintile/1000+/hidden-like/by-wellを出す。
- 1000+はblock全体が1000+となる`min MD >= 1000`へ固定した。
- dedicated tests 12件、py_compile、ruff全check、Jupytext変換/round-trip、strict experiment validationをPASSした。
- `make validate-template`もPASSした。repository全408 testsは405 passed / 1 skipped / 2 failed。
  2 FAILはexp296完了後configと旧test期待の不整合で、exp303変更とは無関係。exp296は変更していない。

## 意図的に未実行

- 大きなKaggle output archive取得（全metricsと生成物SHAをlogで確認できたため不要）
- selector retraining/inference/submission

## 2026-07-21 Kaggle実行承認

- 正規train/inference Notebook placeholderを検証済みcompact self-contained版へ採用する承認を得た。
- push直前の実行契約は`1 fixed readout × 5 evaluation folds / 0 LightGBM config / 0 trained fold / 0 booster / candidate再生成0 / parent再学習0`。
- runtimeはprivate CPU、internet/GPU無効、kernelは`kentookumura/exp303-k-stability-readout-train`へ固定する。
- この承認はreadoutの1回実行だけに適用し、科学的variant追加、selector学習、inference、submissionへ拡張しない。

## 2026-07-21 Kaggle package

- credential checker: OAuth / legacy CLI credential OK。API tokenは未設定だが、CLI実行に必要なOAuth credentialを確認した。
- canonical slugの事前`kernels pull`は403で、同名private kernelが未作成であることを確認した。
- canonical train Notebookを18 cellsのcompact self-contained版へ採用した。inferenceは2 cellsのfail-closed版へ採用したが実行しない。
- package metadata: private=true、CPU、GPU/TPU/internet=false、run_on_push=true。
- kernel id/title: `kentookumura/exp303-k-stability-readout-train` / `exp303 k stability readout train`。
- competition source 1件と固定kernel source 4件をmetadataで確認した。
- package前SHA:
  - config: `9a34f7403bc8b87419c9968d8e7b567451606f2351793153180b514e875191b6`
  - compact train source: `905a9082147e8233364d03b9cdf18327e0075794b212c89edc24a7f33285ea23`
  - canonical train Notebook: `bfd2966f89f5fb3e4f7dce2be8a9e8123b999db846fbfcae12a4defdab1a0855`
  - packaged train Notebook: `7f5090a6fe741c7c363d5c14d6a739e31ebdcf811789ab7e8dcab5f2275196d4`
- package直前検証: dedicated tests 12/12、ruff、py_compile、Jupytext round-trip、strict experiment validation、template validationをPASS。

## 2026-07-21 Kaggle push

- `make push-kaggle-train EXP=exp303_exp226_multiscale_k_stability_selectability_readout_on_exp264`でcanonical slugへ初回pushした。
- Kaggle kernel: `kentookumura/exp303-k-stability-readout-train` version 1、id_no `128080983`。
- pull後metadataでprivate=true、CPU (`machine_shape=None`)、GPU/TPU/internet=false、competition source 1件、kernel source 4件を再確認した。
- Kaggle保存Notebook SHA: `377f02806760a8d1c653074233f1e5ee2587bca7c76f45983f1672eef4979a5f`。
- push承認はversion 1で消費済みに戻し、local config/packageの`run_on_push`をfalseへ戻す。再pushはしない。

## 2026-07-21 Kaggle version 1結果

- 約142.125秒で完了し、最終statusは`technical_pass_scientific_fail`。
- technical checksはfeature coverage、duplicate block 0、truth-before-freeze 0、exp302 prediction SHA、
  corrected exp264 candidate-score SHA、score再計算（最大絶対差0.0）の全てをPASSした。
- pooled H512は7,787 blocks / 773 wells / positive 2,596 blocks。AUC `0.488805`、
  top/bottom positive-rate lift `0.916190x`、mean K16 benefit差 `-1.205532 ft`。
- fold AUCは`0.495293 / 0.449110 / 0.496653 / 0.477773 / 0.520549`で、
  `>0.5`はfold4だけの`1/5`。
- 1000+ / hidden-like spatial / hidden-like typewell-purged AUCは
  `0.488712 / 0.479843 / 0.490046`で、方向PASSは`0/3`。
- 事前登録したpooled AUC、fold方向、quintile positive lift、mean benefit差、subgroup方向は全FAIL。
- 主要SHA:
  - input manifest: `9089d8ded32a7c30ae1504a345993d654a8630ce600ba335a17b1cee99c840ba`
  - feature schema: `ca56361d0aef8a8ffe127418ceadd1cf666dcdeaafd13246a7be19ddfe0e69a7`
  - feature content: `964da0fa966cf24f1f2d3755cf365767f19c37a62c7fab46be819528196a38ea`
  - H512 block content: `55fd6db94a7b4120cde515238e743a273c047aef26f772b32972a4e4cf851267`
  - post-freeze truth content: `c3db553e89c7495cbcc01d99a38ab2b301bacf3e6fc8fa4dffe930eb93b35982`
  - summary JSON: `e7b76c68191273d2798968d5b87e6f1a6ef4cca1401b981e4c74143f1399649e`
- 高instability側ほどK16 benefitが低い逆方向が4/5 foldsと全stress scopeで再現したため、
  threshold不足ではなくfeature familyの不適合と解釈する。
- 事前登録どおり方向反転、feature weight、horizon、boundary幅、thresholdを救済せず、
  selector学習、inference、submissionへ進めない。

## 次のアクション

1. exp303のK-scale instability selector branchは閉じる。
2. 新規救済expは追加せず、独立したexp305 exact-HMM emission auditを優先する。
3. exp303のprediction/inference/submissionは作成しない。
