# exp243_pf_seed_medoids

## 状態

- Route: `pf_beam`
- 状態: v3 full 773 wells exact parity PASS。candidate bankとして支持、direct replacementは不採用
- train-side RMSE: saved/replay `likpf_mean` 11.594898、最良medoid単体 12.296667
- Public LB: なし
- Private LB: なし
- Submit ID: なし
- 作成日: 2026-07-13
- 親実験: `exp072_exp063_full_replay_feature_cache`

## 仮説

likelihood-PFの128 seed平均は複数のplausible trajectory modeを中間値へ潰す可能性がある。
seed trajectoryをtarget-free距離でcluster化し、各modeの実在trajectoryであるmedoidを残せば、
単純平均では得られないblock / whole-well candidate headroomを追加できる。

## 変更点

- exp072のSHA256 modulo + 1 seed契約で、float64の128 seed trajectoryを平均前に保持する。
- tail前半1.0 / 後半1.5 weighted trajectory RMSEを使う。
- 決定的BUILD+PAMで固定K=3/5/8のmedoidを作る。
- exp237 base8 unionへのrow / 128/256/512 block / whole-well oracle追加価値を測る。
- clustering・K・medoid生成にはtrue TVTを使わない。

## 実行契約

- PF replay: 1/well、500 particles × 128 seeds。
- K-medoids postprocess: 3（K=3/5/8、同一PF replayを使用）。
- LightGBM config: 0、fold: 0、booster: 0。
- parent/control再学習: なし。
- Kaggle CPU、internet off、全773 wellsを1 notebookで直列実行。
- raw-test inference / submission: 無効。

## 検証方針

- Fold: なし。exp072互換pseudo-tailの全eligible wellsを1 notebookで監査する。
- Group: well単位で候補とoracleを集計する。
- Stratification: distance bucket、1000+、exp115 hidden-like、worst-well。
- Leakage Check: true TVTはcandidate固定後のscore/oracleだけに使用する。

## 実行入口

- train notebook: `exp243_pf_seed_medoids_train.ipynb`
- inference notebook: `exp243_pf_seed_medoids_inference.ipynb`（disabled contractのみ）
- Kaggle notebook実行を正とし、ローカルnotebook実行は行わない。

## 判定

row oracleだけでなくblock / whole-well oracleが改善し、cluster massとmedoid間距離がmodeとして
解釈可能かを確認する。近重複またはsingleton Monte Carlo noiseだけなら閉じる。selectorと
safety guardが成立するまでraw-test inferenceやsubmitへ進まない。

## 所見

v2の4 shardは3,783,989行 / 773 wellsを完走し、ID/well重複は0だった。ただしsaved exp072
`likpf_mean`とのreplay差はRMSE 0.743077、最大9.847657でexact parity不成立だった。
さらにshard 0/1と2/3でvalidation cacheのsource SHAが異なったため、strict mergeは棄却した。
参考集計でも最良medoid単体は`pf_seed_medoid_k3_m0` RMSE 12.232271で、saved
`likpf_mean`から+0.637373悪化し、worst-well回帰は+18.542614だった。

原因調査で、v2はtypewell TVTと評価MD/Zを`float32`へ丸めてから`float64`へ戻していたことを
特定した。v3はPF入力をCSV由来のfloat64のまま維持し、canonical exp072 v2 cacheのraw /
decompressed SHAとschema SHAを実行前に強制する。saved `likpf_mean`はSHA固定したexp209
enriched cacheから別名復元し、canonical inputへ一対一結合して同名列衝突をなくした。

## v3 full結果

- Kernel: `kentookumura/exp243-pf-seed-medoids-train` version 1、id_no `127058309`
- runtime: 37,067.406秒、3,783,989 rows / 773 wells、全well `ok`
- saved/replay `likpf_mean`: RMSE 11.594898、3,783,989行で差0、exact parity PASS
- 最良direct medoid `pf_seed_medoid_k3_m0`: RMSE 12.296667（+0.701770）、344 wells改善 / 429悪化、worst +20.953998
- exp237 base8 + K8 medoid oracle: row -1.348387、block 128 -1.405104、block 256 -1.372076、block 512 -1.316683、whole-well -1.092839
- all-KはK8単独からwhole-wellでさらに-0.006406だけなので、後続候補はK8に限定する。

## 次

exp243はcandidate generationとして完了する。medoid単体や無条件平均には進まず、保存済みK8候補と
cluster mass / likelihood / entropy / ESS等だけを使うtarget-free selectability auditを別実験で行う。
その信号が成立するまではselector学習、raw-test regeneration、submissionを行わない。
