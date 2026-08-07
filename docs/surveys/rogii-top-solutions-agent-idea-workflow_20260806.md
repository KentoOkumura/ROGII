---
title: ROGII 上位解法と agent-driven Kaggle 着想ワークフロー
date: '2026-08-06'
updated: '2026-08-07'
types:
- survey
- literature_review
- comparison
experiments:
- exp179
- exp182
- exp202
- exp210
- exp212
- exp215
- exp223
- exp235
- exp413
- exp512
topics:
- winning_solution
- agent_workflow
- prompt_design
- skill_design
- candidate_path
- validation_shift
- blind_evaluation
- idea_generation
status: final
summary: 最終上位8解法を比較し、source-hidden blind benchmark 2回でtop 5の15/16・全12案の16/16機構再発見を確認したkaggle-idea-forgeを実装した。
---

# ROGII 上位解法と agent-driven Kaggle 着想ワークフロー

作成日: 2026-08-06

## まず持ち帰るべき5点

1. **point regressionを急がず、posterior・cost volume・candidate pathを保持する。** 1、9、11、13位は2D alignment/cost-volumeをwhole-wellで解き、9位はposterior同士をpoint化する前に融合した。6位は91候補をsoftに融合した。曖昧性を早く1本へ潰さないことが共通している。
2. **多様性をseed数ではなく、情報源・表現・prior・decoderの違いで作る。** typewell、自分自身の既知prefix、空間近傍、PF、HMM、U-Netは異なる失敗をする。単体RMSEだけで候補を捨てず、residual correlation、truth bracketing、oracle coverage、target-free selectabilityを別々に測る。
3. **物理的不変量をaugmentationとsynthetic pretrainingへ使う。** `TVT + Z`を保つ変形、GR gain/offset/drift、厚さ・path warp、現実的に誤ったfirst-pass predictionが上位解法を支えた。単なるノイズ追加ではない。
4. **CVの正しさだけでなく、CVが答えている分布を監査する。** neighbor donor密度、public/privateのwell数、tail well、hiddenのseconds-per-wellまで評価契約に含める。10位はfold-safeなneighbor特徴でもtest geometry shiftで悪化することを示し、7位は小さなPublic差より773-well OOFの順位が正しかったと振り返った。
5. **agent数より、独立探索→role変換→敵対的棄却→portfolio選抜という分業が重要である。** ideatorとjudgeを別contextにし、微小parameter tuningの上限、一失敗の適用範囲、反証条件、計算量を出力契約にする。

## 結論

ROGIIの上位解法は、特定の単一モデルへ収束していない。共通するのは、タスクをrow-wise回帰ではなく**曖昧な軌跡の推論**として扱い、観測・物理prior・候補・不確実性を最後まで残したことである。最も再利用性が高い原理は次の4つである。

- `点予測`から`whole-well posterior / candidate set`への問題表現変更。
- 候補の単体精度ではなく、候補集合のcoverageと誤差非相関性の最適化。
- 物理的不変量を壊さないsynthetic dataと、推論時に実際に遭遇する誤りを再現したpretraining。
- leakage-free CVに加え、hidden分布と実行時間を模したvalidation。

agent運用については、1位が実装の多くをCodex GPT-5.5/5.6で書き、7位がagentのleakage、早すぎる断念、微小摂動、狭すぎる検索を具体的な失敗として報告している。したがって、将来向けの中心改善は「強い一つの万能prompt」ではなく、**事実抽出、失敗範囲の地図化、独立発想、敵対的検証、統合判断を別roleにするskill workflow**である。

初版では設計案までを扱った。2026-08-07追評価では`kaggle-idea-forge`を実装し、上位writeupを発想agentへ見せないsource-hidden benchmarkで2回forward-testした。新規model学習・Kaggle実験・提出は行っていない。

## 調査目的と証拠範囲

### 問い

1. 最終上位writeupには、どのような共通原理と相違点があるか。
2. このリポジトリは有望な信号をどこまで発見し、どの「使い方」で閉じたか。
3. 将来のコンペで、既存案の微小改良に偏らず上位解法級のrepresentation changeを生むには、どのprompt、skill、subagent分業が必要か。

### 取得時点と取得方法

- 取得時点: 2026-08-06 13:07–13:09 UTCを中心に確認。
- Kaggle CLI: v2.2.3、OAuth認証をcredential checkerで確認。
- 最終順位: `kaggle competitions leaderboard rogii-wellbore-geology-prediction --show --page-size 200 --format json`。
- topic一覧: `kaggle competitions topics list rogii-wellbore-geology-prediction --sort-by top|recent`。
- 本文: `kaggle competitions topics show rogii-wellbore-geology-prediction <topic-id>`。
- CLI上の注意: `leaderboard --download`は終了後もPublic leaderboard CSVであり、最終順位の根拠にはならない。また`topics show --format json`は親topic本文を投影しないため、本文取得にはdefault出力を使った。

### 確認できたwriteup

最終leaderboard上位10組と、取得時点のwriteup有無は次の通りである。

| 最終順位 | Team | Private | writeup |
| ---: | --- | ---: | --- |
| 1 | Ruby | 5.639 | [1st Place Solution](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733220) |
| 2 | Bilzard | 5.802 | 未確認 |
| 3 | tereka & Takoi | 5.836 | 未確認 |
| 4 | L & J & A & A | 5.870 | 未確認 |
| 5 | daimaru | 5.940 | 未確認 |
| 6 | k256.dev | 5.984 | [PF, Physics and Row-Level Bagging](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733226) |
| 7 | roglike | 6.057 | [HMM + UNet (agent is all you need)](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733154) |
| 8 | 富士山 | 6.180 | 未確認 |
| 9 | tremors | 6.251 | [9th Place Solution](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733150) |
| 10 | Can | 6.269 | [A Compass, Not a Map](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733315) |

さらに、[11位](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733305)、[13位](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733174)、[14位](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/733201)の詳細writeupを比較対象へ追加した。したがって本稿は**確認できた最終上位8解法**の比較であり、2–5位・8位を代表するものではない。未確認は「存在しない」ことを意味せず、取得時点でtop/recent複数ページとteam/rank検索から発見できなかったという意味である。

### ローカルarchive

- [1位](../discussions/rogii-wellbore-geology-prediction-733220.md)
- [6位](../discussions/rogii-wellbore-geology-prediction-733226.md)
- [7位](../discussions/rogii-wellbore-geology-prediction-733154.md)
- [9位](../discussions/rogii-wellbore-geology-prediction-733150.md)
- [10位](../discussions/rogii-wellbore-geology-prediction-733315.md)
- [11位](../discussions/rogii-wellbore-geology-prediction-733305.md)
- [13位](../discussions/rogii-wellbore-geology-prediction-733174.md)
- [14位](../discussions/rogii-wellbore-geology-prediction-733201.md)

## 上位解法の比較

### 機構別サマリー

| 順位 | 問題表現・主モデル | 観測・候補 | 融合・decode | validation / deployment上の要点 |
| ---: | --- | --- | --- | --- |
| 1 | 345×400の2D alignment、ConvNeXt-small U-Net、column CE + expected-path Huber + GR penalty | calibrated typewell GR、PF heatmap、XY-neighbor、geometry、3 seed×5 fold | default/PF/XY/XY+PF等をweighted ensemble、XY品質gate | CV 4.627 / Public 5.980 / Private 5.639。XYはPublic悪化でもlocal CVを選択 |
| 6 | GPU PF群 + lightweight NN/TCN/curve ensembler | 16 PF config × typewell/self/XY/GR-sim/self-graft等、lag差を含む91候補 | top-k選択ではなくrow-level soft bagging、global de-shrink | CV 5.4577 / Public 5.626 / Private 5.984。strict fold-safe physical prior |
| 7 | HMM + learned emission + second-pass U-Net refiner | 約6000 synthetic wells、現実的に壊したfirst-pass列 | 2 refiner平均、disagreement gateでHMM/neighbor/fuseを必要箇所だけ利用 | OOFはshippable path。Numba/CPU-GPU overlap、約50–65秒/hidden well |
| 9 | whole-well ResNet34 U-Net、autoregressive Conv2D+BiGRU、HMM、segmented beam | cost volume、level/slope state、physics augmentation | HMM×beam/U-Netをposterior productしてからpoint decode、位置band別NNLS | CV約5.28 / Public 5.435 / Private 6.251。3 seed×5 fold |
| 10 | PF/HMM candidate engine、ranking U-Net、LGB selector、TCN/U-Net corrector | `A=TVT+Z`、49 per-well features、candidate bank | member disagreement gate、structural projection | 773 wells / 3.78M rowsでOOF再構築。donor-density shiftを定量化 |
| 11 | whole-well cost-volume U-Net + PF/GBDT | typewellに加え同一well既知prefixをself-referenceとして使用 | well別blend、曖昧時に安全側へhedge、smoothing | CV 5.66 / Public 6.462 / Private 6.299。self-referenceが548 wells中64%でtypewellより近い |
| 13 | 512×768 alignment image、3種2D backbone + 1D leg | 12 channels、物理整合synthetic、cubic-spline trajectory | whole-well posterior decode、decorrelated 1D leg | CV 5.097 / Public 6.008 / Private 6.319。空間・formation情報なし |
| 14 | 30-checkpoint U-Net + 143-feature GBDT + well gate | formation surface constancy、bootstrap seed directions | hierarchical stack、per-well offset/width補正 | CV 6.2518 / Public 6.004 / Private 6.329。promotion gateをbootstrapで判定 |

数値は各writeupの自己報告であり、CV splitや採用submissionが完全に同じとは限らない。横並びの優劣より、どの機構が再現されたかを見るために使う。

### 1位: 2D posterior alignmentを中心に全信号を画像化

1位はTVTを直接回帰せず、horizontal position×typewell depthのalignment probabilityを予測した。targetは正解周辺を指数平滑した分布で、主損失はcolumn-wise cross entropy、補助にexpected pathのHuberとGR-gap penaltyを置く。重要なのはPF予測やXY-neighbor予測も最終値として固定せず、**2D heatmapや距離channelとしてU-Netへ渡した**ことである。

augmentationもtask invarianceに沿う。`TVT + Z`の関係を保つZ-shift、GR affine変換、fault jumpなどを用い、観測だけを壊す一般的なnoise injectionを避けている。最終的にはPF、XY、両方、defaultなど複数モデルを3 seed×5 foldでensembleした。

### 6位: 強い1本より、失敗の異なる91候補

6位はGPU化によりPF探索を約200倍高速化し、長いfixed-lag smoothingと多数の候補生成を実用化した。typewellだけでなく、self-prefix、XY neighbor、GR similarity neighbor、self-graftというreferenceを使い分け、16 configとlag差から91候補を作った。

核心は「候補上位だけを選ぶ」のでなく、弱い候補でもdecorrelatedなら残してsoft fusionしたことである。writeupでは相関0.99程度の追加候補は役に立たず、top-k/hard selectionも悪化したとされる。候補bankでは、単体scoreと集合への限界寄与を分離すべきだと分かる。

### 7位: 推論時の誤りをsyntheticに作ってrefinerへ教える

7位はHMMのfirst-passをU-Netで修正した。ただしreal training wellsに対するfirst-passは過度に正確で、refinerが条件列をcopyする問題が起きた。そこでtruthにanchor付きlow-pass random walkを加え、first-passが現実的に間違う約6000 synthetic wellsを作り、60 epoch pretrain後にrealを8 epochだけfine-tuneした。

もう一つの一般原理はdisagreement gateである。neighbor信号は平均では無価値でも、2 refinerの不一致が大きい箇所だけに作用させると改善した。「弱い信号」を全面blendするか破棄するかの二択にしない設計である。

### 9位: posteriorを融合してからpoint decode

9位はwhole-well U-Net、autoregressive Conv2D+BiGRU、level-slope HMM、segmented beamという異質なdecoderを組み合わせた。特にHMM、beam、U-Netの確率場をproduct fusionし、30% uniform floorを入れてから一点化した。heel-to-toeの位置bandごとにweightを変え、disagreementからspreadも校正した。

この構造は、最終TVT列だけをblendするより多くの情報を保持する。候補・posteriorのどの段階で融合するかを独立の設計変数にすべきである。

### 10位: modelより先に、到達可能性と分布差を測る

10位は`A(MD)=TVT(MD)+Z(MD)`をformation surfaceと捉え、誤差の89%がheel後1500 ft以降にあり、少数wellへ集中すると診断した。per-well polynomialなどのoracleを先に測り、単一直線driftでは目標scoreへ届かないことも確認した。

特に重要なのはneighbor stageの扱いである。CVでは約0.18改善したがPublicでは約0.33悪化した。最寄りtrain well距離がheld-out CVで683 ft、通常train-withinで470 ftとずれ、test geometryが孤立している可能性を示したため最終版から外した。後からPrivateではneighbor版の方が良かった可能性も判明したが、当時の棄却は観測証拠に整合していた。ここから得るべき教訓は「CVを無視する」ことではなく、**CVが本番のdonor densityを再現するかまで設計する**ことである。

### 11位: 同じ信号でも、referenceというroleなら復活する

11位は既知prefixから、そのwell自身のGR-vs-TVT referenceを作った。別wellのtypewellと違い、測定器・calibration・局所地質が一致する。548 evaluable wellsの64%でself-referenceがtypewellより近く、U-Netのmatching imageとPF referenceの両方に使用し、coverage外だけtypewellへfallbackした。

これは、raw self-GRの直接予測が弱くても「reference」「cost-volume channel」「PF emission」としては有効になり得る具体例である。

### 13・14位: 全体画像とwell-level uncertaintyの別解

13位は512×768のalignment imageに12 channelsを構成し、ConvNeXt-T、EfficientNet-B4、HRNet-W18の3画像legと1D legを組み合わせた。空間近傍やformation特徴なしでも上位に入っており、2D formulation自体の強さを裏付ける。syntheticでは`U=TVT+dZ`を使ったdonor transplant、GR gain/offset/drift/missingを扱った。

14位は30 checkpoint U-Net、143-feature GBDT、well-level gateを階層的にstackし、3000-well bootstrapで`gain>=0.01`かつ`P(gain)>0.9`をpromotion条件にした。平均CVだけでなく、well単位の不確実性を採用判断へ使う例である。

## 横断的に支持された原理

### 1. 予測対象は値ではなく「曖昧な軌跡」

1、9、11、13位の2D画像系と6位のcandidate bankは実装が異なるが、複数の妥当なalignmentを保持する点が共通する。GRはdepth ambiguityを持つため、row-wise回帰で局所最適な値を出すより、whole-well smoothnessや構造priorを使って整合したpathを選ぶ方がtaskに合う。

### 2. 候補多様性は独立の最適化目標

seed違いだけでなく、typewell/self/neighbor、PF/HMM/U-Net/beam、causal/smoothed、1D/2Dの違いが効いた。今後のcandidate評価は少なくとも次を分ける。

- 単体RMSE。
- 既存anchor residualとの相関。
- truthを候補範囲が挟む率、top-k oracle、candidate union oracle。
- target-free selectorで選べるか。
- soft fusionしたときのmarginal gain。
- hiddenで再生成できるavailabilityとruntime。

### 3. synthetic dataは量より「何を壊すか」

上位解法のaugmentationは、保存すべき不変量と推論時のfailure modeから逆算されている。7位のfabricated mistakesは「正解らしいfirst-passを少し揺らす」のではなく、実測した誤差分布に近いlow-frequency failureを作った。将来のpromptは、synthetic案に必ず`preserved invariant`と`simulated deployment error`を要求すべきである。

### 4. 正しいCVにも適用範囲がある

well-level GroupKFoldは必要だが十分ではない。neighbor availability、distance、prefix coverage、sequence length、tail horizonがtrain/validation/testで違えば、同じmetricでも別の問いを解いている。Public LBも小標本なら0.1未満の順序を反転し得る。評価はpooled RMSEに加え、paired per-well delta、fold符号、tail、availability/distance bucket、seconds-per-hidden-unitを固定する必要がある。

## リポジトリ実験との対応: 発見できなかったのではなく、roleが違った

以下はhindsightで過去判定を覆すものではない。各実験が閉じた**具体的な実装**と、上位解法が成功させた**別representation/role**を区別するための監査である。

| repoの証拠 | 当時分かったこと | 上位解法との差 | 今後のclosure範囲 |
| --- | --- | --- | --- |
| [exp179](../../experiments/exp179_cnn_sdf_mtp_heatmap_probe/result.md)、[exp182](../../experiments/exp182_cnn_sdf_mtp_heatmap_fullfold_geometry_probe/result.md) | heatmapはtop3 within10が約0.45–0.50で、shuffled/noGRより明確に強かった | winnerはlocal-window heatmap probeでなく、whole-well categorical posteriorをU-Netでdecode | `local heatmap probe`の結果から`2D posterior family`を閉じない |
| [exp202](../../experiments/exp202_heatmap_mdn_candidate_generator_probe/result.md) | 既存PF/Beamへのheatmap top10追加でoracle RMSE 5.0687→2.7455、new-best率0.2525 | coverage headroomは見えていたが、winner6のようなsoft all-candidate fusionへ繋がらなかった | coverage PASSとselectability FAILを別記する |
| [exp210](../../experiments/exp210_heatmap_mdn_full_well_path_generation_probe/result.md)、[exp212](../../experiments/exp212_heatmap_mdn_full_grid_path_generation_probe/result.md)、[exp215](../../experiments/exp215_mtp_full_tail_heatmap_path_generator_probe/result.md) | stitched/learned point path単体が弱く、endpoint extrapolationも大きかった | winner1/9/13はposterior全体をnetwork入力・fusion対象として保持 | `point path generator`が閉じても`posterior feature/fusion`は未棄却 |
| [exp223](../../experiments/exp223_joint_typewell_self_gr_hmm_likelihood_probe/result.md) | self-GR HMM直接値は11.5949→11.3500の弱い改善 | winner11はself-GRをU-Net imageとPF referenceにした | `self-GR direct HMM`と`self-reference family`を分ける |
| [exp235](../../experiments/exp235_fixed_lag_particle_smoother_pf/result.md) | lag64が11.5949→13.4954へ悪化。seed policy等の差もあり、lag128/256は停止 | winner6はGPU PF、異なる設定、より長いlag、全体soft fusionで改善 | confoundedな一実装からsmoother familyを閉じない |
| [exp413](../../experiments/exp413_scale5_likpf_full_replacement_on_exp335/result.md) | PF/HMMをdirect pathでなくML feature bankとして利用しCV 7.8848 / Public 7.201 | 弱い物理予測を別roleへ移す方向は上位原理と整合 | role変換の成功例として残す |
| [exp512](../../experiments/exp512_hjyact_v2_final_10pct_hedge_on_exp413/result.md) | Public 6.541へ到達したがhonest OOF、Private、hidden stochastic determinismは未証明 | 上位writeupはPublic順位反転とhidden runtimeを警告 | Public anchorと科学的anchorを分離する現行判定を維持 |

最大のmissed opportunityはheatmapやself-GRを全く発見しなかったことではない。**弱かった使い方を、そのsignalまたはrepresentation family全体の限界へ拡張しやすかったこと**である。将来はnegative evidenceを次のtupleで保存する。

`(signal, representation, role, fusion, validation regime, compute regime)`

closureは`instantiation closed`、`role closed`、`mechanism closed`の3段階に分ける。mechanism全体を閉じるのは、複数roleの独立検証またはtask invariantとの矛盾がある場合だけとする。

## agent運用についてwriteupが示した事実

### 1位

- 実装の大半をCodex GPT-5.5/5.6で書いたと明記。
- 作者が詳しくなかったPFもAI coding agentsの助けで開発。
- repositoryにはinactive experimentsも多く、agentにコードを説明させる利用を推奨。

### 7位

- public bundleやsplit不一致を見落とすleakageをagentが起こし得る。
- 長いcontextで一度の失敗を一般化し、早く諦める。fresh sessionで再検討する必要があった。
- 明示的に禁止しないと、意味の薄いhyperparameter perturbationへ流れる。
- 長く具体的すぎる検索語で調査範囲が狭まり、人がresources、keywords、papersを渡す必要があった。
- U-NetはPublic 7–8の構成では効かなかったが、5–6の構成では大きく効いた。methodの有効性は周辺pipelineに依存する。

これらは単一参加者の経験であり、すべてのagentに普遍的とは断定しない。ただしこのリポジトリのheatmap/self-GR/smootherの履歴と同じfailure modeを指しており、workflow設計へ取り込む価値が高い。

## 将来コンペ向けのmaster prompt

OpenAI公式の[Prompting](https://learn.chatgpt.com/docs/prompting)と[Best practices](https://learn.chatgpt.com/guides/best-practices)は、Goal、Context、Constraints、Done whenを具体化することを勧める。[Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)はboundedで独立したread-heavy taskによるcontext分離、[Build skills](https://learn.chatgpt.com/docs/build-skills)はfocusedな責務と明確な入出力を推奨する。これをKaggle着想へ落としたprompt案が以下である。

```text
あなたは Kaggle 研究責任者です。目的は実装開始ではなく、現在の証拠から
「異なる原理に基づく、反証可能な次実験候補」を発見し、portfolioにすることです。

Goal:
- [COMPETITION] で上位解法につながり得るidea cardを8–12件作る。
- top 5はrepresentation change、異なる情報源、fusion/uncertainty、
  validation/deploymentのうち3 family以上を覆う。
- pure hyperparameter tweakは最大2件、top 5では最大1件とする。

Context:
- rules/data/metric: [PATHS OR URLS]
- evidence cutoff: [DATE/COMMIT]
- route別anchorとtrusted CV: [ANCHORS/CV CONTRACT]
- compute/deadline/submission budget: [BUDGET]
- external data/model rules: [RULES]
- saved OOF/models/features: [ASSET INDEX]

Constraints:
- AGENTS.mdとrepo-local skillsに従う。
- 事実・推論・仮説を分離し、各主張にsourceを付ける。
- cutoff後の結果、Public LB、公開test固有artifactをground truthにしない。
- 一失敗からsignal family全体を閉じない。negative evidenceを
  (signal, representation, role, fusion, validation, compute)で記録する。
- 各signalを predictor / candidate / posterior / feature / prior /
  augmentation / uncertainty gate / postprocess のroleへ展開する。
- candidate案ではcoverage/oracleとtarget-free selectabilityを分ける。
- 実装、push、submissionはしない。結果を変える選択はユーザーへ返す。

Workflow:
1. task、availability、leakage、runtime contractを1ページに圧縮する。
2. 独立subagent A–Cを起動し、全員の完了を待つ。
   A: top-solution mechanism archaeologist
   B: repo failure / reusable-asset cartographer
   C: related-domain mechanism researcher
3. A–Cの要約だけを渡し、互いの案を見ないideator D–Fを新contextで起動する。
   D: representation / invariance inventor
   E: candidate diversity / fusion inventor
   F: validation shift / deployment / compute inventor
   各agentは最低6案。parameter-onlyは禁止。
4. 全案を匿名化・deduplicateし、signal×role matrixの空白を確認する。
5. 別contextのcritic G–Iが壊す。
   G: leakage / hidden-test adversary
   H: statistical falsifier
   I: runtime / reproducibility engineer
6. hard gate通過案だけを採点し、main agentがtop 5 portfolioを作る。
   多数決で一案へ潰さず、成立条件、counterevidence、非相関性を残す。
7. top 3に cheap proxy → full OOF → inference smoke の段階試験を作る。
8. final前にsource、kill criterion、hidden compatibility、computeを再監査する。

Idea card:
- id / one-line idea / mechanism family / role
- source facts / hypothesis / preserved invariants
- closest past experiment and exact difference
- counterevidence / expected failure mode
- cheap proxy / full validation / fixed subgroups
- coverage test / selectability test
- runtime and memory / inference contract
- kill criterion / reopen criterion
- rubric score / evidence confidence

Done when:
- 8–12 distinct cards、top 5で3 mechanism families以上。
- 全案にsource、counterevidence、falsification、computeがある。
- unsafe/leaky案は採点前に除外。
- ユーザーがtop 1–3を選べる比較表になっている。
```

## subagentの分業

### Wave 1: 事実を圧縮する

| Role | 読むもの | 読まないもの | 出力 |
| --- | --- | --- | --- |
| Solution archaeologist | 上位writeup、notebook、公式資料 | repo backlog | source claim cards、mechanism matrix |
| Failure cartographer | cutoff時点のrepo、result、OOF manifest | 後のwriteup | exact closure tuple、未検証role、再利用asset |
| Domain researcher | metric、data schema、関連分野の一次資料 | 他agent案 | transferable mechanism、前提、risk |

Solution archaeologistには提案をさせず、`task formulation / target / representation / information / candidate / fusion / augmentation / validation / compute / failure`へ分解させる。Failure cartographerにも順位付けをさせない。この分離により、既存backlogと有名解法へのanchoringを減らす。

### Wave 2: 独立発想する

3 ideatorは互いの案を見ず、task cardとWave 1の圧縮結果だけを読む。最低条件は次の通り。

- representation inventor: 2案以上は回帰以外、2案以上はinvariance/augmentation、1案以上はsynthetic/pretraining。
- bridge/diversity inventor: predictor、posterior、feature、reference、candidate、augmentation、gate、fusionの全roleを走査。
- validation/deployment inventor: distribution shift、tail、runtime、memory、offline inferenceを手法の一部として提案。

### Wave 3: 発案と独立したcriticが壊す

- leakage critic: fold leakage、same-OOF selection、public artifact、train-only input、hidden cardinalityを検査。
- statistical critic: paired per-well、fold variance、tail、availability/distance bucket、powerを検査。
- runtime critic: seconds-per-unit、model load、peak RAM/VRAM、determinism、fallbackを検査。

criticは「familyが駄目」と返さず、どのtupleがどの条件で棄却されるかを返す。main agentだけがportfolioを統合し、書き込み担当も一つに限定する。

## skill設計と実装状況

既存の[kaggle-survey-papers](../../.agents/skills/kaggle-survey-papers/SKILL.md)、[kaggle-strategy](../../.agents/skills/kaggle-strategy/SKILL.md)、[kaggle-review-exp](../../.agents/skills/kaggle-review-exp/SKILL.md)はそれぞれ外部調査、戦略選択、実験契約に強い。一方、writeupの機構抽出と独立発想の間が明示的な責務になっていない。巨大な一skillへ統合せず、次の小さいskillを提案する。

| Skill | 単一責務 | 必須入力 | 主出力 | 停止条件 |
| --- | --- | --- | --- | --- |
| `kaggle-solution-distill` | writeupを再利用可能な機構へ分解 | competition、rank/URL、cutoff | claim cards、mechanism matrix、source ledger | rank/source未確認、本文欠落、fact/inference混在 |
| `kaggle-failure-cartography` | negative resultの適用範囲を限定 | repo snapshot、result/metrics、asset index | closure tuples、未検証role、reopen条件 | exact baselineや変更点を復元できない |
| `kaggle-idea-forge` | role-separated agentから独立idea portfolioを作る | task card、claim cards、failure ledger、budget | idea cards、signal×role matrix、unranked portfolio | CV/availability/compute contract欠落 |
| `kaggle-idea-eval` | idea workflow自体をsnapshot回帰試験 | frozen snapshots、gold mechanisms | recall、安全性、多様性、attribution report | cutoff後sourceを隔離できない |

このうち[kaggle-idea-forge](../../.agents/skills/kaggle-idea-forge/SKILL.md)を2026-08-07に実装した。JSON schema、構造validator、source-hidden境界、negative closure tuple、task-first発想、same-entity context、imperfect-intermediate training、invariant discovery、candidate diversity、adversarial gateを一つのfocused workflowにした。`kaggle-solution-distill`、`kaggle-failure-cartography`、`kaggle-idea-eval`は本稿時点では独立skillとして未実装である。

既存skillの改修候補は次の通り。

- `kaggle-survey-papers`: sourceごとの文章要約だけでなく、共通schemaのclaim cardを出せるようにする。
- `kaggle-strategy`: idea cardを入力とし、safe / high-upside / compute-enablerのportfolio枠を別々に残す。
- `kaggle-review-exp`: 選択cardの`idea_id`、preserved invariant、kill/reopen criterionをsteeringへ引き継ぐ。

`AGENTS.md`には全作業で不変のルールだけを置き、上記の反復workflowとoutput schemaはskillへ置く。custom promptだけに依存せず、代表的なactivation testとfailure caseをskillに添える。

## ideaのhard gateと採点rubric

以下は上位writeupから観測したscoreではなく、**将来運用向けの提案**である。

### 採点前hard gate

1. rulesと推論時data availabilityを満たす。
2. fold-safeでhidden testに再構成可能。
3. hypothesisと変更mechanismが識別可能。
4. cheap proxyとfull testの両方がある。
5. runtime/memory上限へ到達する道筋がある。
6. sourceとcounterevidenceがある。

### 100点rubric

| 項目 | 点 |
| --- | ---: |
| task fit / invariantとの整合 | 15 |
| 期待upside / oracle headroom | 15 |
| anchorとの誤差非相関性 | 12 |
| representation / information / roleの新規性 | 10 |
| 証拠の強さ | 10 |
| 反証可能性・診断価値 | 10 |
| hidden分布を模すvalidation | 10 |
| tail/fold安定性 | 6 |
| compute・iteration speed・unlock価値 | 5 |
| inference再現性・offline適合 | 5 |
| 実装量あたり情報利得 | 2 |

点数とは別にevidence confidenceをA/B/Cで持つ。低確信・高upside案を平均点で消さず、portfolioのexploration枠へ残す。

## prompt / skillの回帰試験

上位解法を読ませて要約できるだけでは、将来の着想能力を測っていない。次の2種類を分ける。

### Source-visible translation test

- 1位から2D posterior、soft target、PF heatmap、physics augmentationを別mechanism cardとして抽出できる。
- 6位から「PFを増やす」ではなく、異なるreference×decorrelated candidates×soft fusionを抽出できる。
- 7位からfabricated-error pretraining、disagreement gate、seconds-per-hidden-wellを抽出できる。
- 10位からoracle parameterization、retrieval failure、donor-density shiftを抽出できる。
- 11位からself-GRをimage/PF/fallbackの複数roleへ分けられる。

### Source-hidden rediscovery test

cutoff時点のrepo snapshotだけを渡し、後のwriteupと結果を見せずに確認する。

| Test | 期待するtop-5内の発想 |
| --- | --- |
| Representation reset | row regression以外に2D cost-volume / posterior path prediction |
| Heatmap role recovery | point pathが弱くてもfull posteriorをmodel input/fusionへ使う |
| Self-reference role recovery | self-GR directが弱くてもreference/image/PF emissionを未棄却にする |
| Candidate diversity | residual correlationとbracketingで候補を選びsoft fusionする |
| Smoothing + compute | fixed-lag/full smootherと高速化を一組で提案する |
| Invariance augmentation | `TVT+Z`保持、GR affine、path/thickness warp、realistic residual corruption |
| Validation shift | donor-distance matched CVとavailability bucketを要求する |
| Hidden execution | publicのwell数・行数を固定せず、seconds-per-wellを見積もる |
| LB skepticism | exp512 Public 6.541を科学的anchorへ自動昇格しない |
| Closure scope | 一つのdirect/point実装FAILからfamily全体を閉じない |

提案する合格目標は、mechanism recall@5 80%以上、unsafe promotion 0、source attribution precision 95%以上、top 5で3 family以上、pure parameter tweakはtop 5中1件以下である。単一agent、同一roleのmulti-agent、role分離のみ、role分離+criticの4条件を、同じsource bundle・model・reasoning effortで最低3回比較する。source-hidden testではinternetとcutoff後ファイルを遮断しなければ、着想でなくretrievalを測ることになる。

## Source-hidden blind benchmark実行結果

### 情報隔離

2026-07-12をcutoffとする[task packet](../../studies/kaggle_idea_forge_benchmark/rogii_source_hidden_packet_v1.md)を作った。内容は公式task/deployment contractと、当時存在したE01–E07の実験結果だけである。最終writeup、順位、後続exp413/512、現在のsurvey、gold mechanism名を含めていない。

- control: fresh agentへpacketだけを渡した。
- treatment v1: 別のfresh agentへ同じpacketと`kaggle-idea-forge` v1だけを渡した。
- treatment v2: v1の欠落からskillをgenericに改修し、互いの出力を見ないfresh agentで2回実行した。
- 発想agentはWeb、repo探索、writeup、survey、後続実験、他run出力を禁止した。各runは別ファイルへ保存した。
- judgeは発想完了後に別contextで起動し、初めて8件の上位一次archiveとgold mechanismを読んだ。

静的な出力だけでunauthorized readが絶対になかったことまでは証明できない。ただしjudgeは、順位、Private score、91候補、具体epoch数、上位固有の実装名など、packet外fingerprintの混入を検出しなかった。

### Round 1

| 条件 | top 5 | 全12案 | unsafe | parameter-only | coverage/selectability |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 10/16 | 15/16 | 0/12 | 0/12 | 12/12分離 |
| skill v1 | 11/16 | 15/16 | 0/12 | 0/12 | 12/12分離 |

v1はtop 5のfamily diversityとmulti-family soft fusionを改善したが、発想集合全体はcontrolと同点だった。最大の欠落は、現実的に壊したfirst-pass conditioningからpretrainし、real OOF-like入力で短くfine-tuneするrefinerが一枚の実験として接続されなかったことである。詳細は[Round 1 judge](../../studies/kaggle_idea_forge_benchmark/round1_judgement.md)を正とする。

### v2で変更したこと

- 詳細な失敗履歴より先にtask cardだけから案を作る`task_first` pass。
- 同一entityの部分観測をentity-specific reference/conditionerへ変え、coverage外fallbackを要求するprobe。
- OOFで中間予測の誤差過程を測り、corruption pretrain→real OOF-like fine-tune→held-group testまでを一枚へ閉じるprobe。
- data schemaからinvariant候補を出し、EDAで支持された関係だけをaugmentation/target/constraintへ使うprobe。
- 候補多様性をobservation/reference/representation/dynamics/decoder/seedに分解する規則。
- top 5をrepresentation、information、data generation、candidate/fusion、validation/computeの5枠に分けるschema v2。

ROGII固有語、勝者model、`TVT+Z`等のgold answerはskill本文へ入れていない。

### Round 2

| 条件 | top 5 | 全12案 | G7 corrupted-conditioning refiner | safety | validator |
| --- | ---: | ---: | ---: | --- | --- |
| skill v2 run 1 | 15/16 | 16/16 | 2/2 | PASS | schema v2 PASS |
| skill v2 run 2 | 15/16 | 16/16 | 2/2 | PASS | schema v2 PASS |

両runは独立に次をtop 5へ生成した。

1. whole-well increment distributionを保持し、prefix境界からjoint decodeする表現変更。
2. same-well prefix GR/TVTをentity-specific reference/calibrationへ変え、短prefix・範囲外・曖昧時はglobal referenceへfallbackする案。
3. OOF candidateのbias、自己相関、欠損、rank inversionを再現したconditioningでpretrainし、real OOF-like bundleでfine-tuneするrefiner。
4. heterogeneous candidatesへsoft posteriorを置き、disagreement/coverage不確実時はML anchorへabstainする融合。
5. observation cache、batch/streaming、path parityを使い、上記algorithmのfull OOFとhidden実行を解禁するcompute enabler。

上位解法との対応はG1 whole-group probability/coherent decode、G2 weak posterior as evidence、G3 same-entity reference/fallback、G4 role-diverse candidates/soft fusion、G5 uncertainty gate、G7 corrupted first-pass refiner、G8 compute/runtimeで各2/2だった。G6 domain-invariant syntheticはtop 5では1/2だが、全12の独立augmentation案で2/2となった。詳細なcard対応、安全性、実装時のnested-selection注意は[Round 2 judge](../../studies/kaggle_idea_forge_benchmark/round2_judgement.md)を正とする。

### 判定の範囲

本結果から言えるのは、**ROGIIの上位中心機構を、writeupを見せず、pre-writeup証拠から反証可能な実験案として再発見できた**ことである。実際にその案を学習すれば同じscoreまたは順位になることは証明していない。またskill設計自体がROGIIから得た抽象原理を使っているため、別コンペへの一般化は未証明である。ここから先に同じROGII goldへ合わせ続けるとbenchmark overfitになるため、次の正しい評価先は別コンペのfrozen snapshotである。

## failure mode / anti-pattern

- 長いmain contextを全agentへ複製し、全員を同じbacklogへanchorする。
- 独立案生成前にdebateし、majority consensusへ収束する。
- ideatorとjudgeを同じagent/contextにする。
- 上位解法のモデル名だけをコピーし、問題表現や不変量を抽出しない。
- direct predictorの一失敗でsignal familyを閉じる。
- point pathが弱いことでposterior/heatmapも閉じる。
- 単体RMSEだけで非相関候補を捨てる。
- oracle coverageが高いだけでselectorを承認する。
- pure hyperparameter tweakを新規ideaとして水増しする。
- Public LBの小差をvalidationと扱う。
- GroupKFoldだけでneighbor density等のhidden geometry差を見ない。
- public stub全体時間をhidden runtimeとみなす。
- 全agentに同じファイルを並行編集させる。
- ROGIIのhindsightへ過適合し、将来コンペでも常にU-Net/PFを推す。

## 解釈

### 事実として支持されたこと

- 2D whole-well alignment、candidate/posterior-level fusion、self-reference、physics-consistent synthetic dataは複数の上位解法で独立に使われた。
- PublicとPrivateの順位は大きく変動し、6位はPublic 20位、11位はPublicで約1240位相当からPrivate 11位になった。
- hidden runtimeと本番availabilityは精度と同じく最終解法を制約した。
- AI coding agentsは少なくとも1位と7位の開発へ大きく関与した。

### 本稿の推論

- このリポジトリの主な問題は着想の完全な欠如ではなく、negative resultのscopeとsignal roleの管理不足だった可能性が高い。
- future workflowではagentを増やすだけでなく、独立context、非対称role、共通output schema、adversarial gateが必要である。
- candidate bankはモデル性能だけでなく、情報資産としての多様性を管理対象にすべきである。

### 未解決

- 未公開の2–5位・8位writeupが共通原理を補強または反証する可能性がある。
- 1位のrepositoryや全公開notebookを再実行した再現監査は行っていない。
- ROGIIではsource-hidden再発見を確認したが、別コンペのheld-out frozen snapshotへ一般化するかは未評価である。
- どのsubagent数・context量・reasoning effortが費用対効果最良かは未検証である。
- 生成案を実装した場合のCV/LB改善は未検証であり、mechanism recallと実スコア改善を混同しない。

## 関連ファイル

- writeup archive: [1位](../discussions/rogii-wellbore-geology-prediction-733220.md)、[6位](../discussions/rogii-wellbore-geology-prediction-733226.md)、[7位](../discussions/rogii-wellbore-geology-prediction-733154.md)、[9位](../discussions/rogii-wellbore-geology-prediction-733150.md)、[10位](../discussions/rogii-wellbore-geology-prediction-733315.md)、[11位](../discussions/rogii-wellbore-geology-prediction-733305.md)、[13位](../discussions/rogii-wellbore-geology-prediction-733174.md)、[14位](../discussions/rogii-wellbore-geology-prediction-733201.md)
- 既存統合監査: [exp238 selector / TVT feature audit](exp238_selector_tvt_feature_audit_20260716.md)、[candidate path blend audit](candidate_path_blend_audit_20260716.md)、[GR matching deep research](gr_matching_deep_research_20260625.md)
- 現行方向: [KAGGLE_DIRECTION.md](../../KAGGLE_DIRECTION.md)、[experiment_summary.md](../../experiment_summary.md)
- 実装skill: [kaggle-idea-forge](../../.agents/skills/kaggle-idea-forge/SKILL.md)、[schema](../../.agents/skills/kaggle-idea-forge/references/portfolio-schema.md)、[validator](../../.agents/skills/kaggle-idea-forge/scripts/validate_portfolio.py)
- blind artifacts: [packet](../../studies/kaggle_idea_forge_benchmark/rogii_source_hidden_packet_v1.md)、[control](../../studies/kaggle_idea_forge_benchmark/control_run1.md)、[v1 treatment](../../studies/kaggle_idea_forge_benchmark/treatment_run1.md)、[v2 run 1](../../studies/kaggle_idea_forge_benchmark/treatment_v2_run1.md)、[v2 run 2](../../studies/kaggle_idea_forge_benchmark/treatment_v2_run2.md)、[Round 1 judge](../../studies/kaggle_idea_forge_benchmark/round1_judgement.md)、[Round 2 judge](../../studies/kaggle_idea_forge_benchmark/round2_judgement.md)

## 次のアクション

1. 2–5位・8位のwriteupが公開されたら、同じreportを更新し、別reportを乱立させない。
2. 同じROGII goldへの追加prompt tuningは止め、別コンペのpre-writeup frozen snapshotでtransferを評価する。
3. 新コンペ開始時は、baseline実装前のtask-first portfolioと、複数実験後のevidence-inversion portfolioを別々に保存する。
4. 生成cardを実験化するときは`kaggle-strategy`で選択し、`kaggle-review-exp`へ`idea_id`、invariant、cheap/full gate、kill/reopen criterionを引き継ぐ。
