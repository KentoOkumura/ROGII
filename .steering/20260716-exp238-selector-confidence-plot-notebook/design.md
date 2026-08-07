# 設計

## アプローチ

exp083 v12 の all-well TVT plot を基礎に、exp148 OOF の代わりに exp238 final
`lgb_mean_pred_tvt` を表示する。exp238 selector train の5個の nested score gzip は
chunk 読みし、各 outer fold の `role=valid` 行だけを `row_index` へ配置する。

selector は candidate ごとの予測絶対誤差を出すため、row ごとの score 最小 candidate を
top-1、2番目との差を confidence margin とする。TVT panel には top-1 candidate path を
破線で重ね、下段には margin line と candidate category strip を置く。plot title と manifest
には dominant top-1 candidate と share を保存する。

配色は exp083 v12 を正とする。true TVTは黒、ML OOFは`#e11d48`、PF ANCCは
`#1f77b4`、Beamは`#ff7f0e`、LikPFは`#2ca02c`、exp226は`#a16207`、
exp209 HMMは`#7c3aed`に固定する。selector top-1 pathは候補系列との誤認を
避けるため exp083 のlast-anchorと同じ`#64748b`の灰色破線とし、候補の識別は
top-1色帯で行う。exp083にない candidate は stable Tableau colors を固定する。

## 実験範囲

- 対象実験: `exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218`
- Route: `ml_model`
- 親実験: exp238 final train v5 / selector train v4、可視化参照は exp083 v12
- 変更する変数: diagnostic notebook と plot/manifest/summary 出力のみ
- 固定する変数: exp238 OOF、outer-valid selector score、11 candidate、fold/SHA contract、final prediction

## 再現性設計

- seed policy: 新規 RNG は使わない。入力は保存済み OOF/candidate/selector score に固定する。
- stochastic 処理の有無: notebook 内にはない。PF/Beam 候補は upstream の固定 train cache を読む。
- PF/Beam / likelihood-PF / seed bagging の有無: 可視化入力としてのみ存在し、再生成しない。
- 並列処理と乱数の関係: 並列処理と乱数は使わない。
- CPU/GPU runtime と deterministic flags: CPU、internet disabled。model fit と GPU 処理は0。
- train cache / test feature regeneration の SHA 記録方針: selector summary が宣言する5 scoreの decompressed SHA、入力 file path、row/well/coverage を summary に保存する。test regeneration はない。
- model manifest / prediction / submission SHA 記録方針: model/submission は生成しない。exp238 OOF input SHA と生成 manifest/summary/plots zip SHA を保存する。
- Kaggle package bootstrap 確認方針: self-contained notebook と metadata だけの custom package とし、CPU/internet off、必要 kernel sources を静的確認する。実行依頼後は同じ canonical kernel で `run_on_push=true` とし、version を追加する。

## リスク

- リークリスク: selector top-1 を可視化する際に outer-train score を混ぜると診断が楽観化する。`role=valid` と row coverage を fail-fast する。
- CV/LB 不一致リスク: top-1 candidate は exp238 final prediction ではなく、selector safety guard も worst-well で不通過。図と summary にこの caveat を明示する。
- ランタイム/メモリリスク: 5個の約378万row score gzipを走査する。chunk 読みし、11列 score matrix は valid row の top1/marginへ即時縮約する。
- 再現性リスク: upstream candidate cache の更新で同名file内容が変わり得る。実行時の入力 SHA と selector summary の固定 score SHA を保存する。
