# exp243_pf_seed_medoids 結果

## 状態

v2 Kaggle CPU 4 shardはreplay parityとinput SHAのstrict merge gateが不通過だった。
原因を特定したv3は1 well probeに続き、full 3,783,989行でもexact parityを確認した。
candidate bankとしての仮説は支持、direct replacementは不採用として完了する。

## 仮説

PFの128 seed平均で失われる複数trajectory modeを実在seed medoidとして保持すると、
`likpf_mean`およびexp237 base8 candidate unionへblock / whole-well headroomを追加できる。

## 固定設定

- Route: `pf_beam`
- PF: exp072互換raw-GR Gaussian likelihood、500 particles × 128 seeds、1 replay/well
- seed: exp072 stable SHA256 modulo + 1 per well + seed index
- dtype: trajectory / replay mean / K-medoidsはfloat64、保存列はfloat32
- distance: tail前半1.0 / 後半1.5 weighted trajectory RMSE
- clustering: deterministic BUILD+PAM、K=3/5/8
- LightGBM / fold / booster: 0 / 0 / 0
- inference / submission: 無効

## 評価項目

- direct candidate RMSE、1000+、hidden-like、worst-well
- row、128/256/512-row block、whole-well oracle
- exp237 base8 union oracle差
- unique-best率、cluster mass、entropy/HHI、within/between distance
- saved exp072 `likpf_mean`とのreplay parity

## 旧v2結果

v2の4 shardは合計3,783,989行 / 773 wellsを完走した。shard別well数は
188 / 196 / 193 / 196、全well statusは`ok`、IDとwellの重複はともに0だった。

ただしsaved exp072 `likpf_mean`とのreplay parityは成立しなかった。

- mean absolute difference: 0.373289
- difference RMSE: 0.743077
- max absolute difference: 9.847657
- `1e-6`超の差: 3,714,240 / 3,783,989行

schema SHAは4 shardで一致したが、validation sourceのdecompressed SHAはshard 0/1が
`0503de05...536`、shard 2/3が`99a3c70a...350`に分かれた。このためstrict mergeを
成立扱いにはしない。

参考値として十分統計を全行加重集計すると、saved `likpf_mean`はRMSE 11.594898、
replay meanは11.601992（+0.007094）、最良medoid単体`pf_seed_medoid_k3_m0`は
12.232271（+0.637373）だった。最良medoid単体は338 wells改善 / 435悪化、最大回帰
+18.542614でdirect candidate guardも不通過だった。

exp237 base8との参考oracle差はall-K unionでrow -1.104800、block 128 -1.169740、
block 256 -1.138025、block 512 -1.095331、whole-well -0.961581だった。ただしparityと
provenanceが不成立なので、selector headroomの採用根拠には使わない。

## v3修正

- `numeric_array()`のfloat32変換をPF入力へ使わず、typewell TVTと評価MD/Zを直接float64で読む。
- canonical exp072 v2 cacheをraw SHA `14faee3a...f18`、decompressed SHA
  `99a3c70a...350`で固定する。
- schema SHAを`700d3814...b8`で固定する。
- canonical exp072 cacheには`likpf_mean`がないため、saved parity controlはSHA固定したexp209 enriched
  cacheから`likpf_mean_exp209_reconstructed`として復元し、ID一対一確認後に結合する。
- `fba7683c`ではv2のfloat32経由sigma `22.658950811666866`がKaggle記録
  `22.658950811666863`と一致し、直接float64では`22.658358259724082`になることを確認した。
- full shard packageと407-row parity probe packageを分離し、bootstrap内config/helperの一致を確認した。
- probe v1はcanonical cacheに`likpf_mean`を要求してPF前に停止。科学計算を変えず、正しい別control
  契約へ直したv2を同じkernel IDへpushした。

## v3 parity probe結果

- Kernel: `kentookumura/exp243-pf-seed-medoids-parity-probe-v1` version 2
- 対象: `fba7683c`、407 rows / 1 well
- runtime: 7.003974秒（PF処理完了ログは6.6秒）
- seed base: 787424823
- GR sigma: 22.65835825972408
- mean absolute difference: 0.0
- difference RMSE: 0.0
- max absolute difference: 0.0
- `1e-6`超の差: 0 / 407行
- exact parity: PASS
- canonical exp072 input、schema、exp209 parity controlのSHAはすべて期待値と一致。

## v3 full結果

- Kernel: `kentookumura/exp243-pf-seed-medoids-train` version 1、id_no `127058309`
- Kaggle status: `KernelWorkerStatus.COMPLETE`
- PF audit runtime: 37,067.406秒（約10時間18分）
- coverage: 3,783,989 rows / 773 wells、全773 well status `ok`
- canonical exp072 input raw/decompressed SHA、schema SHA、exp209 parity control raw/decompressed SHAは全て期待値と一致
- downloaded `metrics.json` SHA: `693efe12ab9d379a5afe321d31a6ba8d28c38e85f50e3f529d0101d08f4cfc89`

saved exp072 `likpf_mean`とv3 replay meanは、全行で厳密に一致した。

- mean absolute difference: 0.0
- difference RMSE: 0.0
- max absolute difference: 0.0
- `1e-6`超の差: 0 / 3,783,989行
- saved/replay RMSE: 11.594897672

最良direct medoidは`pf_seed_medoid_k3_m0`だが、RMSE 12.296667365でbaseから
+0.701769693悪化した。well別は344改善 / 429悪化、median delta +0.054901、worst regression
+20.953998。全distance bucketで悪化し、1000+は+0.769445、hidden-like spatial / typewell-purgedも
+0.705972 / +0.616778だった。したがってmedoid単体のdirect replacementは不採用。

一方、exp237 base8へK8 medoidを候補追加したoracle改善は全scopeで維持された。

| scope | base8 | base8 + K8 | delta |
| --- | ---: | ---: | ---: |
| row | 4.564605 | 3.216218 | -1.348387 |
| block 128 | 4.805040 | 3.399936 | -1.405104 |
| block 256 | 4.883135 | 3.511059 | -1.372076 |
| block 512 | 5.036480 | 3.719798 | -1.316683 |
| whole-well | 6.592426 | 5.499587 | -1.092839 |

K8 medoidはbase8+K8 union内で1,660,416行、43.8800%のunique-bestを持ち、whole-well
oracleでは374 / 773 wellsを改善した。all-Kのwhole-well RMSEは5.493181で、K8単独からの追加改善は
0.006406に留まる。cluster summaryもmedian max-pairwise distance 15.603104、normalized entropy
0.891478で、近重複や単一cluster collapseだけではない。

## 判定と次

exp243の「平均前modeを実在medoid候補として保持する」仮説は支持する。ただしtarget-freeな選択規則は
未確立なので、候補生成物としてのみ採用し、direct replacement・平均・raw-test inference・submissionは
行わない。後続は保存済みK8候補に限定し、cluster mass、likelihood mass/gap、assignment distance、entropy、
ESS/resampling、base path disagreementのtarget-free selectability auditを先に行う。ここで信号が成立した場合だけ、
別実験でouter-well fold selectorを検討する。
