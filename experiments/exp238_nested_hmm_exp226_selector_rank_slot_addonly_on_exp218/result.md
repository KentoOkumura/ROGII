# 結果

## 結論

selector CPU notebook v3と、ユーザー承認によるfinal GPU notebook v5がともに完了した。strict nested OOFの`lgb_mean`は7.936690で、保存済みexp218 8.475794より名目上0.539104改善した。ただしexp238はselector生成物に保存されたouter foldを正としており、GPU runtimeで再構成したGroupKFoldおよびhistorical exp218のfold割当と一致しない。このため改善量をrank-slot追加特徴だけの効果とは断定しない。

さらにselector単体のworst-well最大回帰は+37.680897でguard不通過のままである。nested OOF自体は有効なtrain-side evidenceだが、自動promotion条件は満たさない。ユーザーがこのリスクを確認して明示承認し、hidden-safe修正後のcode submission ref `54662073`はPublic LB `7.775`で完了した。これはexp218 `7.843`を改善するため、リスク注記を維持したままML route submitted anchorへ採用する。

## 実行

- Kernel: `kentookumura/exp238-nested-selector-train` v3
- Runtime: 14,049.055秒（約3時間54分）
- rows / wells: 3,783,989 / 773
- candidates: 11
- selector context features: 184
- selector: outer 5 × inner 4 = 20 boosters
- candidate-long上限: train/valid各120,000行/model

## Safety readout

| 指標 | selector - fallback RMSE |
| --- | ---: |
| global | -3.089911 |
| near `000_050` | -0.609540 |
| `1000_plus` | -3.372225 |
| worst-well最大回帰 | +37.680897 |

global、near、long-tailはいずれも改善したが、well単位の極端な回帰が残った。既定上限`+0.25`を大幅に超えるため`guard_pass=false`は妥当である。

## Final GPU train

- Kernel: `kentookumura/exp238-nested-rank-slot-exp218-train` v5
- Runtime: 15,635.579秒（約4時間20分）
- rows / wells: 3,783,989 / 773
- features: exp218 base 380 + nested selector 35 = 415
- LightGBM: 3 configs × 5 folds = 15 boosters
- parent/control再学習: なし

| model | pooled RMSE TVT |
| --- | ---: |
| lgb0 | 7.972601 |
| lgb1 | 7.971170 |
| lgb2 | 7.990319 |
| lgb_mean | **7.936690** |

`nsel_top1_minus_anchor`はlgb0平均importance 3,547.6で最上位、`nsel_top2_minus_anchor`も2,823.0で上位に入り、nested selector特徴がモデルに利用されたことは確認できる。

## 比較上の制約

- selector生成物の`row_index + role`を正のouter fold contractとして使用した。
- selector CPU環境とGPU runtimeで再構成したGroupKFoldは5 foldすべて一致しなかった。
- historical exp218 controlは別fold割当で学習されているため、`7.936690 - 8.475794 = -0.539104`は参考差であり、同一foldの因果的ablationではない。
- nested stackingではouter-valid wellの正解TVT由来rank/errorをvalidation特徴へ流しておらず、同一fold target leakageは認められない。

## 判断

GPU本学習は完了し、nested OOFは有望な数値を示した。ただしworst-well guard不通過とfold非一致による比較不能が同時に残る。推論化するならユーザーがこのリスクを明示的に受容する必要がある。特徴効果を厳密に確かめる場合は、同じartifact outer folds上のexp218 base-only controlが必要だが、追加GPU 15 boostersとなるため別途承認なしには実行しない。

## Fold-matched inference

- selector train v4でouter 5 × inner 4の20モデル本体を保存し、v3 nested score SHAとの完全一致を確認した。
- selector inference v3は学習0本で保存済み20モデルを適用し、outer fold別に5つのraw-test score面を生成した。
- final inference v2は保存済み15 LightGBMへ同じouter foldのselector score面を対応させ、14,151行を予測した。
- prediction範囲は11,590.633〜12,240.465、平均11,904.923402、標準偏差278.717954。
- `submission.csv`はsample submissionと行数・列・ID順が一致し、重複・欠損・非有限値はない。
- submission SHA: `dc0eb2e8f4581d0e91a8a6748f13cae17742e86539cbc234fa3a42fad6ec1f9d`。
- v2監査時点ではcompetition提出未実行だった。後続の提出結果とhidden-safe修正は次節に記録する。

## Code submission失敗とhidden-safe修正

- ユーザー提出ref `54647064`（notebook version id `334800917`）はhidden rerunで未処理例外となり、scoreは付かなかった。
- 原因はpublic test 14,151行用のexp145 rawtest特徴、selector score面、exp073 cache、exp226 submissionをhidden testへ適用していたことにある。
- 修正版は学習済み20 selectorと15 final LightGBMだけを固定入力とし、行依存のbase replay、HMM、exp226 K16、multiobs、exp145 learned likelihood、GRWRをcurrent testから再生成する。
- selector inferenceは同じ提出notebook内でouter別4モデル平均をメモリ内生成する。selector再学習とfinal LightGBM再学習はどちらも行わない。
- hidden-safe inference v3は407.557秒で完走し、14,151行、415特徴、保存済み20 selector、保存済み15 final LightGBMの契約を満たした。selector・final modelの再学習は0本。
- code submission ref `54662073`、scriptVersionId `334897917`はerrorなしで`COMPLETE`、Public LB `7.775`となった。

## Public LB判断

- exp238: `7.775`
- prior ML anchor exp218: `7.843`（exp238が-0.068改善）
- exp148 CPU runtime: `7.921`（exp238が-0.146改善）
- ensemble anchor exp082: `7.601`（exp238は+0.174劣る）

したがってexp238をML routeの新しい提出anchorへ更新する。ただしensemble routeの全体最良はexp082のままである。selector worst-well guard不通過とhistorical exp218とのfold不一致は、LB改善によって解消されたわけではない。

## Raw-test copcf parity v1

- Kernel: `kentookumura/exp238-rawtest-copcf-parity` v1、id_no `127304223`
- Runtime: CPU、ログ最大時刻560.479秒、internet disabled
- current test: 14,151行 / 3 wells
- selector context: trainと同じ184列。missing列0、全行nonfinite列0、部分nonfinite列0、nonfinite値0
- `copcf_*`: 41/41列にfinite値があり、visible testでは全41列が全行finite
- exp226診断: 4/4列が全行finite
- selector: 保存済みouter 5 × inner 4の20 modelを読込。学習0本
- score面: outer別5面、各14,151行、numeric nonfinite値0
- test-test edge / neighbor: 不使用
- final LightGBM学習・推論、submission生成・competition submit: いずれも未実行

受け入れ基準の「184 context」「41 copcf」「exp226診断4列」「missing context 0」および
5 score面の契約をすべて満たした。従来の45列全行NaNはtestデータが3 wellsだから不可避だった
のではなく、test側generator不足による問題であり、full-train referenceだけから生成できることを
実行で確認した。test wellは独立変換で、hidden testのwell数が増えてもtest-test関係は使用しない。

主要SHA:

- context decompressed: `91d65d9c86fa9b83f9bcff5ddf509812620c20259865c64af2d9fd3f137f6d48`
- schema: `5a67fb13d30191bf4cce674ed5ac76cfdb64b2fc6a5c29eb9c16ae4462b99f69`
- loaded selector manifest: `7b58aba498dfc1e2e507557f07b95fd7dff88cf5fbd260c086dbdbd6bf749908`
- score outer0..4 decompressed: `16e4e4b34c92966d57fe2232207bb4a71f6f0b30765d99e949c7c9f0445af6b0`,
  `1b152399be3723c442c37e17470cdf9d468ec00ba13eef42a56194dac4d5d1e5`,
  `600cb78f3f4c2dc842ee5a6fc7077cf54c88ccd85f2ea1460d4159d37baa9d56`,
  `e1db0f48eb2a5a0a4a348e93d160f35686e3fd60fce4d3e034f55e8a3f329e78`,
  `b0cdaf6c917feb8ba97d67bac09d6163da95bac5114b6ae6fcd55614a626f7f6`

この監査はNaN/parity問題だけを解決したもので、exp238のadd-only設計やselector worst-well
guard不通過は解消していない。次段ではこのgeneratorを正規current-test推論へ組み込み、別途
outer-fold safeなgated/bounded direct readoutを評価する。

## Copcf parity final inference実装

parity通過済み184 context generatorを、保存済み20 selectorと15 final LightGBMへ接続した。
既存正規inferenceは上書きせず、`*_inference_copcf_parity.py/ipynb`を別名で作成した。
outerごとのinner 4 selector平均から35 rank-slot特徴を作り、同じouterの3 final modelへ渡す。
最終schemaはexp218 380 + selector 35 = 415列である。学習はselector/final/controlすべて0、
competition submitは行わない。Jupytext、py_compile、ruff、strict validation、関連pytest
10件、Kaggle bootstrap/config検証はpassした。Kaggle実行結果はまだない。

## Copcf parity final inference v1

- Kernel: `kentookumura/exp238-copcf-parity-inference` v1、id_no `127309057`
- Status: `COMPLETE` / `hidden_safe_copcf_parity_final_inference_completed_not_submitted`
- Runtime: summary 479.031秒、T4、internet disabled
- current test: 14,151行 / 3 wells
- parity: 184 context、41 `copcf_*`、exp226診断4列、missing/部分nonfinite列0、fallback 0
- model: 保存済みselector 20本、保存済みfinal LightGBM 15本、学習0本
- schema: exp218 base 380 + selector 35 = 415列
- prediction: min 11,591.132、max 12,240.252、mean 11,904.986795
- submit-check: sampleと行数・列・ID順が一致し、重複・欠損・非有限値なし。PASS
- submission SHA: `c1a16392519e14f2b4ca9c1d86668e7f13d0f7bc20088c165f0aedcec6b05d30`
- prediction decompressed SHA: `d88e9ca83197267b0d749953a0fa9ff506e3a2c2a0ddb6f796bf13c84f0f5fec`
- context decompressed SHA: `6cde0ad35e9fd4e91d0c6ccbb1a117caa2f8d99113317409cfec22316b28c8fe`
- selector surface decompressed SHA: `a7b317035eced14c55190b624332e1059aeda4f0a3993abaea77b487332b4c56`

NaN/parity修正を含む最終推論はKaggle上で正常完走し、提出形式も検証済みである。competition
submitはまだ行っていない。selector worst-well guard不通過とadd-only設計は維持されているため、
この結果は推論実装の健全性を証明するが、モデル品質guardを新たに通過したことは意味しない。

## Copcf parity final inference v1提出

2026-07-15にkernel `kentookumura/exp238-copcf-parity-inference` v1をcode submissionした。
submission refは`54725625`、submit直後statusは`PENDING`。提出SHAは
`c1a16392519e14f2b4ca9c1d86668e7f13d0f7bc20088c165f0aedcec6b05d30`で、提出直前検証は
FAIL/WARN 0だった。Public LBは未確定であるため、parity修正前ref `54662073`の7.775を
この提出結果としては扱わない。継続監視は行わない。

## Copcf parity final inference v1 Public LB

submission ref `54725625`は`COMPLETE`となり、Public LBは`7.842`だった。parity修正前のhidden-safe v3 ref `54662073` / `7.775`より`+0.067`悪化し、exp218 ref `54457577` / `7.843`とは`-0.001`のほぼ同等である。

したがって、test側で184 contextをfinite生成するNaN/parity修正は実装上正しいが、その有限値化は精度改善にはつながらなかった。exp238のML route anchorはref `54662073` / `7.775`を維持し、parity版ref `54725625`は採用しない。

## OOF selector-confidence diagnostic v1

- Kernel: `kentookumura/exp238-oof-selector-confidence-probe` v1、id_no `127444478`
- Status: `COMPLETE` / `diagnostic_plots_completed_not_submitted`
- Runtime: CPU、ログ最大時刻656.56秒（約11分）、internet disabled
- 対象: 3,783,989行 / 773 wells。outer 0〜4の`role=valid`は各行を1回ずつ覆い、欠損・重複なし
- 出力: 773 / 773 well plots、plot manifest、selector top-1 distribution、plots zip、summary JSON
- 学習・候補再生・submission生成・competition submit: すべて0

| global metric | value |
| --- | ---: |
| exp238 `lgb_mean` OOF RMSE | **7.936690** |
| selector top-1 hard path RMSE | 8.512264 |
| Likelihood PF mean RMSE | 11.594898 |
| exp226 K16 RMSE | 9.427110 |
| selector confidence margin mean / p50 / p90 | 0.319253 / 0.156284 / 0.714208 |

selectorが最も高く信頼した候補は`Self-GR HMM`で、1,205,794行（31.8657%）だった。次いで`PF ANCC` 20.3814%、`exp226 K16` 17.5745%、`Likelihood PF mean` 12.8689%、`LikPF/HMM 50:50` 8.3958%である。各plotには青のexp238 OOF、橙破線のselector top-1 path、top2−top1 margin、11候補の色分け帯を表示した。

hard top-1はexp238 OOFよりRMSEが0.575574悪く、historical worst-well guardも`+37.680897`で不通過のままである。したがってこの出力はselectorの信頼先を観察するdiagnosticとして使い、exp238 final predictionの置換やanchor更新には使わない。

## OOF selector-confidence diagnostic v2配色修正

ユーザー指示により、v1の配色を参照元exp083 v12と同じ配色へ修正した。同じcanonical kernel `kentookumura/exp238-oof-selector-confidence-probe` v2は`COMPLETE`、CPU / internet disabled、ログ最大時刻938.857秒（約15分39秒）で完走した。

| series | color |
| --- | --- |
| true TVT | `black` |
| exp238 ML OOF | `#e11d48` |
| selector top-1 diagnostic | `#64748b`（灰色破線） |
| PF ANCC | `#1f77b4` |
| Beam mean | `#ff7f0e` |
| Likelihood PF mean | `#2ca02c` |
| exp226 K16 | `#a16207` |
| exp209 HMM / band | `#7c3aed` / `#8b5cf6` |

selector top-1色帯のPF ANCC、Beam、LikPF、exp226、Self-GR HMMも同じcandidate色に揃えた。代表図`000d7d20.png`を目視確認し、上記配色、灰色破線のtop-1 path、候補色帯と凡例が正常に表示されることを確認した。

変更は描画色とsummary内の`plot_colors` contractのみで、3,783,989行 / 773 wells、RMSE、selector top-1 distribution、strict outer-valid coverageはv1と完全に一致した。773 / 773 plotsを再生成し、学習・submissionは行っていない。今後の参照はv2を正とする。

## OOF Likelihood-PF 128 paths diagnostic v1

ユーザー指定により、exp072 likelihood-PFを500 particles × 128 stable seedsで全773 wellsについて再生し、各wellの128 trajectoryをすべて重ねた。Kaggle kernel `kentookumura/exp238-oof-likpf-128-paths-probe` v1は`COMPLETE`、CPU / internet disabled、summary runtimeは14,067.881秒（約3時間54分28秒）だった。

| global metric | value |
| --- | ---: |
| rows / wells / plots | 3,783,989 / 773 / 773 |
| exp238 `lgb_mean` OOF RMSE | **7.936690** |
| Likelihood-PF 128-seed mean RMSE | 11.594898 |
| saved PF mean exact parity wells | 773 / 773 |
| saved PF mean max abs difference | 0.0 |

manifestは773行でwell/pathともunique、各wellが128 seeds・1 seedあたり500 particlesだった。青の128本はalpha 0.06 / linewidth 0.55、true TVTは黒、exp238 LGB OOFはroseの不透明線で描画した。代表図`000d7d20.png`を目視し、128-path分布、truth、LGB OOFが同一図に正常表示されることを確認した。142 wellsは描画だけ最大6,000点へ間引いたが、parityとRMSEは全行で計算している。

このrunはPFのseed不確実性を観察するdiagnosticであり、model fit、prediction blend、submission生成、competition submitは行っていない。PF平均はLGB OOFよりRMSEで3.658208悪いため、直接置換やanchor更新の根拠にはしない。

## OOF selector-confidence diagnostic v3共通typewell順

ユーザー指定の共通typewell順を反映した同じcanonical kernel
`kentookumura/exp238-oof-selector-confidence-probe` v3は`COMPLETE`、CPU / internet disabled、
ログ最大時刻925.583秒（約15分26秒）で完走した。3,783,989行 / 773 wellsを処理し、
773 / 773 PNGをexp065の`native_overlap` / threshold `0.999`による54 typewell groupsへ
並べた。

- PNG名は`typewell_{typewell_order:04d}_{well}.png`で、先頭は
  `typewell_0001_09441b8d.png`、末尾は`typewell_0054_f5859199.png`。
- manifest 773行をexp065対応表と全列照合し、group順、group内well順、filename、coverage、
  unique性が一致した。
- Kaggle output APIの773 PNGが辞書順でtypewell順になることを確認した。
- 343,358,561 bytesのplots zipは全体を取得せずcentral directoryをrange取得し、773 membersが
  manifest順と完全一致することを確認した。
- 先頭・末尾PNGはともに2093×1485で、先頭図を目視して線、色、凡例、軸が正常であることを
  確認した。

global RMSE、selector confidence、selector top-1 distribution、outer-valid coverage、v2で固定した
配色はすべて不変だった。学習、candidate再生成、prediction変更、submission生成、competition
submitはいずれも行っていない。この変更はdiagnostic出力の並び順だけであり、exp238の評価、
ML route anchor、戦略バックログは変更しない。今後のselector-confidence plot参照はv3を正とする。
