# exp514_exp413_likpf_seed_bank_reuse_on_exp512 結果

## 状態

Stage A fixed32 technical / determinism gateはKaggle version 1でPASS。Stage B version 2は正常完走したが、
shared-bank SP45 candidateが固定all-AND精度gateをFAILした。Stage Cは不要。Stage D version 3は
submission-ready経路を完走し、v2と5出力SHA完全一致、200-well上限推定`7.957332時間`でruntime estimated
PASSだったが、Stage Bを代替できないためexp514は不採用・提出不可として終端する。
その後ユーザーがStage D v3をcode submissionしたref `55266559`はhidden rerunの未処理例外でscoreなしとなった。
Stage D v4はOOM対策とhidden-safe SHA guardを実装し、Kaggle visible technical validationを完走した。
5出力SHAと提出形式はPASSし、200-well推定は`6.289658〜8.057147時間`。再提出は行っていない。

## 仮説

exp413 stable-seed likelihood-PF bankをSP45とexp413で共有すれば、exp413 scale5を維持しつつ
SP45重複PFを除去でき、200-well推論の9時間制限へ近づけられる。

## 固定設定

- 親: `exp512_hjyact_v2_final_10pct_hedge_on_exp413`
- Route: `ensemble`
- PF: x1.0、500 particles、128 seeds、scale 3/5/8/12
- seed: well別SHA256 stable base
- 学習: 0 model / 0 fold / 0 booster
- 精度監査: Stage Aと同じtarget-free fixed32 paired replay（小規模screening）
- runtime監査: Stage D visible testの工程別実行時間から200 wellsへ外挿

## 結果

| メトリック | 値 |
| --- | --- |
| fixed32 technical | PASS（32 wells、thread 1/4、各2 runで3 SHA完全一致） |
| fixed32 paired RMSE | FAIL（control 9.010759、candidate 9.060440、delta +0.049680 ft） |
| Stage C 200-well shadow | 不要（ユーザー指示、PASS扱いではない） |
| Stage D visible runtime | v4 COMPLETE（3 wells / 14,151行、879.386897秒） |
| 200-well estimated runtime | 6.289658〜8.057147時間、estimated PASS（hidden実測ではない） |
| Public LB | error / scoreなし（ref 55266559） |

## Stage D code submission ERROR

- ref `55266559`、scriptVersionId `340328874`、submitted at
  `2026-08-05 10:05:07.813 UTC`。Kaggle APIは`COMPLETE`かつscore空、hidden rerun unhandled errorを返した。
- hidden tracebackは取得不能。ただしStage D末尾のv2 output-equivalence 5 SHA検査がhiddenでも無条件に走り、
  visible固定SHAとの不一致で必ず例外になるhidden-incompatible defectを確認した。
- visible v4 parent peak RSS `25,123.293 MiB`はchild processを除き、Ridge解放前のpeakも保持するため、hidden OOMは
  未証明のままである。Stage D v4でmemory lifetimeとSHA guardを修正しvisible実測したが、再提出は行っていない。

## Stage D version 4 OOM修正

- Ridge終了後にtrain/OOF/model/matrixを解放し、shared PF開始前の常駐メモリを削減した。
- shared PFとSP45を4-thread well streamingへ統合した。128-seed raw pathと全scale full payloadは最大4 wells
  だけ同時保持し、SP45終了直後に破棄する。exp413まで保持するのは2列float32だけである。
- SP45→HJYACT→exp413を全行deep copyではなく所有権移譲に変更し、exp413内のcaller-side copyと
  予測DataFrameの不要なreturnを除去した。
- visible v2の5出力SHA検査をvisible sampleだけに限定し、hiddenでは`SKIPPED_HIDDEN_DYNAMIC`とする。
- source `ff668d88...542d`、Notebook `15d4ebeb...1316`、19 contract tests、Ruff F821、Jupytext、
  strict validationはPASS。Kaggle version 4も`COMPLETE`し、5出力SHAはv2と完全一致した。
- Ridge current RSSは`12,979.207→791.668 MiB`、exp413前は`953.895→865.621 MiB`、shared bank終了時は
  `1,359.234→1,037.676 MiB`へ低下。raw bankは保持せず、visible full payloadは最大3 wellsだった。
- visible totalは`879.386897秒`でv3より`39.690306秒`（`4.727%`）遅い。200-well外挿は
  `6.289658〜8.057147時間`でestimated PASS。submit-checkもPASSしたが、hidden runtime/OOM保証ではない。

## Stage B version 1 ERROR

- 129,906行×16列のcontrol/candidate predictionはtruth/fold join前にfreeze済み。
- freezeは`5,081.673128秒`、ERRORは`5,083.549175秒`。
- pre-branch列を既存post-branch列名へrenameして同名列が2本でき、shape `(129906,2)`と
  target `(129906,)`のbroadcastに失敗した。
- 性能gateは一度も計算されていないため、科学的なPASS/FAILには分類しない。

## Stage B version 2修正

- ユーザー承認により、評価関数`metric_bundle`をcopy + 1次元配列の明示代入へ修正した。
- 変更対象はpost-freeze採点処理だけで、32 wells、予測経路、selector、branch hedge、gate閾値は不変。
- version 2のKaggle再実行前に、重複列を作らない回帰test、構文、Ruff、Jupytext、strict validationを行う。
- 上記検証は全てPASSし、同じprivate T4 / internet off kernelへversion 2をpushした。

## Stage B version 2結果

- Kaggle version 2は`COMPLETE`。本体runtime `4,845.475189秒`、32 wells / 129,906 rows。
- v1/v2のfrozen prediction content SHAは`62c78373...8a280`で同一。truth/foldはfreeze後にjoinされ、
  source SHAも固定値と一致したため、採点修正による科学predictionの変更はない。
- primary pooled deltaは`+0.049680497 ft`で上限`+0.02 ft`をFAIL。
- nonworse foldは`2/5`で必要な`4/5`をFAIL。fixed scopeはraw-GR observedが`+0.060618254 ft`で
  上限`+0.05 ft`をFAIL。by-well p95は`+0.647871442 ft`で上限`+0.25 ft`をFAILした。
- worst well `+1.192163546 ft <= +5.0 ft`、全fold nonempty、全scope nonemptyはPASS。
  hidden-like 2面はともに`-0.117664920 ft`改善したが、固定all-ANDは`FAIL`。
- 事前規約に従い、同じ32 wellsでscale/seed/selectorを調整する救済は行わない。Stage D v3の
  runtime/readiness確認は有効な実装証拠として残すが、exp514 candidateは提出しない。

## Stage D version 1 ERROR

- 3 visible wells / 14,151行のHjyact component生成とsubmission auditまでは約780秒で完了した。
- Hjyact componentはID順一致・finiteだが、SHA `6b3e1c57...37b3`が親exp512固定SHA
  `b192d3f3...ed4a`と異なり、log `783.969秒`で停止した。
- exp514はSP45 bankを科学的に変更する候補なので、これは速度FAILではなく、親output exact parity guardを
  そのまま継承したreadiness実装エラー。exp413 component、最終50/50、正式runtime外挿は未実行。

## Stage D v1による200-well暫定評価

- v1実測4工程 + 親exp512の未実行exp413 proxy（shared PF短縮を反映）: `8.448〜9.831時間`。
- shared PF短縮をproxyへ反映しない保守推定: `9.129〜10.739時間`。
- いずれも上限が9時間を超えるため、暫定gateは`estimated_fail`。exp413工程未実測のため不確実性はhigh。
- v2では親exact SHAを要求せず、v1で得たexp514候補SHAをvisible witnessとして検証する。
- 修正・静的検証後、同じprivate T4 kernelへversion 2をpushし、`COMPLETE`を確認した。

## Stage D version 2結果とversion 3最適化

- v2はshared PF `39.7344秒`、SP45 `1.085秒`、HJYACT `69.610950秒`、Gold逐次
  `123.116097秒`、exp413 `274.684秒`、固定overhead `421.699344秒`で完走した。
- 200 wellsへの固定式外挿はlower `8.068150時間`、upper `9.528814時間`。upper基準で
  `estimated_fail`であり、visible 3 wellsからの推定なのでhidden完走時間の保証ではない。
- v3はGoldを最大4 processのwell並列にし、SP45決定論featureとimputerをHJYACTへ再利用する。
  HJYACT固有stochastic featureだけを再生成し、科学設定と最終式は不変。
- v2のGold/HJYACT/exp413/component readout/final submissionの5 SHAをv3に固定した。
  どれか1つでも不一致ならruntimeが短くてもFAILとする。
- v3は5 SHAすべて完全一致でPASS。Goldはvisible 3 wellsを3 processで実行し`37.392%`短縮、
  HJYACTは決定論176列を共有して`44.883%`短縮、全体は`90.233199秒`（`9.703%`）短縮した。
- 200-well外挿はlower `6.174531時間`、upper `7.957332時間`。上限基準で
  `estimated_pass_not_hidden_runtime_guarantee`へ改善した。
- submissionは14,151行、sample ID順完全一致、duplicate/NaN/Inf 0でsubmit-check `PASS`。
  外部competition submitは実施していない。

## 実装結果

- 親exp512 source SHA、exp073 replay source SHA、exp413 config SHAのdriftがないことを確認した。
- compact self-contained inference候補は8,155行 / 53 cells、Stage A専用候補は746行 / 7 cells。
- SP45はprecomputed shared bankだけを読み、legacy `run_pf_lik_ensemble_scales`を呼ばない。
- exp413はshared scale5 / arithmetic mean frameだけを読み、後段`replay_source.build_likpf`を呼ばない。
- learned x1.3、Gold masked-prefix PF、`pf_ancc`、`pf_z`、Beamは親経路のまま残した。
- 16 contract tests、構文、Ruff F821、Jupytext round-trip、strict `validate-exp`がPASSした。
- Stage A専用Notebookをprivate Kaggle T4 / internet offで実行した。正規Notebook、full prediction、
  submissionは作成していない。

## Stage A結果

- Kernel: `kentookumura/exp514-shared-likpf-fixed32-stage-a` version 1、id_no `129757357`。
- 実行時間: report `2,363.410299秒`、kernel log上のreport出力時刻 `2,378.627551秒`。
- 32 wells、500 particles、128 seeds、thread 1/4、各2 rerun、合計128 well-bank生成。
- thread 1の2 runは`892.045002 / 885.807420秒`、thread 4は`282.509160 / 290.578054秒`。
- aggregate SHA `68c5dc68...c8e74a`、branch SHA `904a3e00...6d0285`、ledger SHA
  `5a3a81f8...a42942`は4 runすべて完全一致した。
- truth readなし、new booster 0、parent/control retraining 0、submission file 0、external submit 0。
- exp413 scale5は、独立した二重PF実行ではなく、SHA固定したexp073 x1.0 sourceとのAST/source契約一致と
  adapter/ledger fail-closeを根拠にexact contractを確認した。

## 再現性

- deterministic anchor: false
- candidate source SHA: `961762731f91bf20de6d43d869aeed44bfa98f60be7f8cccc1c65b37d05dc24c`
- Stage A source SHA: `89129ad85c129145e635633741e08ff5e058a365c344a4a4bdbdc77190ab3873`
- Stage A aggregate / branch / ledger SHA: 上記のとおり生成・4 run一致
- full prediction / submission SHA: 未生成
- Kaggle kernel version: 1
- Stage A report SHA: `87387d9a...7b8612`

## 解釈

Stage Aにより、実Numba経路がthread scheduleとrerunに対して決定的で、candidate内の共有bank・adapter・ledger契約が
fixed32で動作することは確認できた。Stage D v3のvisible工程別外挿では200-well上限が9時間未満になった一方、
Stage BはSP45 legacy bankをexp413 stable-seed bankへ置換するとpooled・fold・raw-GR observed・by-well p95を
同時に維持できないことを示した。したがって高速化は成立しても科学的置換は不採用である。Stage Aの39.4分は
4 passのPF単体監査なのでruntime見積もりには使わない。
Stage D visible testでは、4-way並列工程を`工程秒×200/4`から`工程秒×200/visible wells`、逐次工程を
`工程秒×200/visible wells`で外挿し、固定overheadを1回だけ加える。これは高不確実性の見積もりで、hidden実測ではない。

## 次

exp514はStage B scientific FAILとcode submission hidden rerun ERRORで終端し、Stage C、追加救済、再提出へ進まない。
将来のruntime改善はlegacy SP45 stochastic bankの科学条件を維持した実装最適化に限定し、exp413 bankとの
科学的共通化を再利用しない。
