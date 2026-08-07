# Source-hidden idea portfolio: control run 1

## 評価境界

- Evidence cutoff: **2026-07-12**
- 許可された競技・実験情報源: **`studies/kaggle_idea_forge_benchmark/rogii_source_hidden_packet_v1.md` のみ**
- Web、Kaggle、writeup、survey、後続実験、backlog、他 run の出力は使用していない。
- 本文中で packet に明記されていない事項は、事実ではなく仮定または検証対象として記す。
- 実装、学習、Kaggle 実行、提出は行っていない。

## Task card

1. **予測対象と損失**: 各 horizontal well の未知 suffix の全行について `tvt` を予測する。row-pooled RMSE なので、少数 well の長く持続する大誤差が支配的になり得る。
2. **推論時に既知**: horizontal well の measured depth / trajectory coordinates / GR / prefix の `TVT_input`、paired typewell の depth-indexed reference（GR を含む）、runtime の `sample_submission.csv`。未知なのは suffix の TVT と train-only formation columns。
3. **依存単位**: row は well 内 sequence として依存し、typewell と horizontal well は paired reference を作る。selection、blend、adapter は unseen well に外挿する必要がある。
4. **現出力が捨てるもの**: ML anchor は point prediction、既存 path は top-1 または固定 blend、heatmap は局所候補の分布を持つ。候補間 disagreement、heatmap の多峰性、候補集合の oracle headroom、well-level の持続誤差状態が point 化で失われている。
5. **保存すべき invariant**: prefix との TVT 連続性、trajectory に沿う順序と滑らかさ、同一 well 内の長距離整合、推論時に存在する列だけの利用、well 完全分離の OOF、決定的再生成。
6. **hidden で変わる条件**: well 数・行数・suffix 長・GR 分布・候補の disagreement・typewell availability の細部は変わり得る。public 3 wells の ID、行数、SHA、分布には依存しない。約 200 wells、offline、9時間、CPU/T4、動的列挙・sample ID への一対一整列を contract とする。

## 明示的な missing / assumptions

- train で本番同等の prefix/suffix mask が提供されるか、複数の擬似 cutoff を作る必要があるかは packet だけでは不明。以下では **各 train well から leakage なく本番相当 mask を再現できる**ことを仮定し、できなければ prefix-adaptation 系の confidence を下げる。
- 既存5 path と ML anchor の予測が既知 prefix 上にも定義できるかは不明。I01/I08 ではこれを cheap test の最初の availability check にする。
- hidden の総行数・1 well 当たり最大行数・peak-memory 上限は不明。runtime 判定は well 数だけでなく row 数と worst-well 長に対して外挿する。
- 「意味のある改善幅」は指定されていない。予測案の full OOF では、暫定的な事前登録基準として **anchor 9.5264 から 0.10 以上の pooled RMSE 改善**を採用する。これは packet 由来の事実ではなく、微小な fold noise を追わないための研究上の閾値である。

## Negative-evidence closure ledger

| Evidence | 閉じた具体案 | 閉じていない範囲 | 理由 |
|---|---|---|---|
| E03 | tested local heatmap をそのまま direct replacement にする実装、大 window の当該実装、tested geometry channel の top-3 改善仮説 | heatmap logits を候補 scorer、feature、uncertainty、局所 branch proposal として使う role | real GR と shuffled/no-GR の差が大きく、top-10 coverage 0.8089 は signal 自体の存在を支持する一方、worst-well coverage 0 と oracle RMSE は単独 point 出力を支持しない。 |
| E04 | truth-aware oracle union を deployable selector とみなすこと | union の coverage headroom と、別途学習する target-free selectability | existing 5 + heatmap 10 の oracle 2.7455 は表現余地を示すが、truth-free selector は未証明。両者を必ず別評価する。 |
| E05 | stitched-only point paths、full-grid extrapolation、probability-weighted full-tail point path の tested implementations | 強い existing backbone を保持した局所 branch、heatmap を path evidence にする role、constrained hybrid lattice | learned paths 単独は非常に悪いが、union oracle は一貫して改善し、rank 1/2 の逆転例は主に calibration / ranking failure を示す。 |
| E06 | tested additive likelihood の raw inference、強 weight、既存の direct candidate / hard switch / dense gate、10h50m CPU path | revealed prefix TVT を使う低次元 correction、弱 signal を bounded conditioner にすること、O(KN) path-only scorer | 弱 weight は全距離 bucket を改善したが well 間の異質性と worst regression が大きい。signal family 全体ではなく role と compute regime を限定して閉じる。 |
| E07 | 500-particle・128-seed・lag-64 ancestor smoother の当該実装 | smoothing 単独の因果効果、seed-paired ablation | seed namespace が control と異なり、unsmoothed tail まで悪化したため因果が未同定。ただし runtime と明確な悪化から、本 portfolio では優先しない。 |

## 12 idea cards

### I01 — Revealed-prefix low-rank path warp

- **Mechanism / hypothesis**: ML anchor と既存5 path のそれぞれについて、既知 prefix 上の `truth - prediction` を `offset + bounded slope (+ 高々1 hinge)` の低次元 warp として fit し、suffix へ shrinkage extrapolation する。持続する well-level miss が boundary 以前から観測できるなら、モデル再学習なしに長い bias を除ける。
- **根拠 / 反証**: E01 の「長く相関した誤差」が支持。E06 は same-well prefix signal に情報がある可能性を示すが、461 improve / 312 worsen と +46.95 ft worst regression は単純な強補正に反対する。prefix 誤差と suffix 誤差の関係自体は packet では未確認。
- **最も近い失敗との差**: E06 は prefix **GR** を HMM likelihood / switch / gate に使った。I01 は inference で明示的に revealed な prefix **TVT label** に対する candidate residual だけを使い、自由度を数個に制限し、補正量を OOF quantile 内へ shrink する。候補選択はしない。
- **Cheap test**: cached OOF prediction が prefix 上にもあるかを確認し、全 well で本番相当 mask を再生。degree 0/1/hinge の nested well-GroupKFold を行い、候補別に suffix RMSE、1000-ft-plus bucket、worst 20 wells、補正量を比較する。最初は offset-only で十分な情報性があるかを見る。
- **Full validation**: outer whole-well 5-fold の各 held-out well に対し、train folds だけで warp family・regularization・clip を決定。ML anchor 単体、各 path、固定 blend、anchor へ shrink した最終予測を評価する。
- **Kill criterion**: prefix residual と suffix の平均 residual の held-out 相関が実質ゼロ、または最良の事前固定 warp が anchor 比 0.10 未満の改善、または pooled gain と引き換えに worst-20-well RMSE を 5%以上悪化させるなら終了。candidate path が prefix 上に未定義なら、そのままでは終了し I08 にも availability risk を伝える。
- **Coverage / selectability**: candidate coverage は一切増えない。selectability も解かず、各既存 point path の calibration だけを変えるため、候補選択効果と混同しない。
- **Hidden runtime / inference**: well ごとの小行列 fit と1回の vector 補正で O(N rows)、CPU、追加 memory は数本の vector。formation columns、public ID、suffix truth は不使用。dynamic well 列挙、sample-ID alignment、補正係数・clip の deterministic serialization を必須にする。
- **Family / confidence**: representation + information role change / **B**。

### I02 — Heatmap-as-likelihood path rescoring

- **Mechanism / hypothesis**: heatmap の top-1 path を出さず、各 existing whole-well path が各 local window のどの logit mass を通るかを sample し、entropy・margin・window coverage で信頼度を付けて well-level compatibility score に集約する。OOF listwise calibrator が existing paths（必要なら anchor を含む）の soft weight を出す。heatmap の genuine GR signal は「候補を作る」より「既存 path を評価する」方が calibration error に強いはず。
- **根拠 / 反証**: E03 の real/shuffled/no-GR 差、E04 の added-candidate oracle headroom、E05 の rank 1 が97 ft、rank 2が1.2 ftの例が支持。E03 の worst-well top-3=0、local top-10 oracle 13.2963、E05 の stitched-only 46–50 は heatmap evidence を無条件に信頼することへ反対。
- **最も近い失敗との差**: E05 は heatmap mode を stitched/full-tail **point path** として出した。I02 は strong existing path の座標で heatmap logits を読む **observation likelihood** であり、新しい learned point pathも右端 extrapolation も作らない。E06 の hard switch と違い、weight は well-OOF の soft calibration と anchor shrinkageを持つ。
- **Cheap test**: 保存済み5-fold logits/ranks/entropy と existing five OOF paths だけで、候補ごとの mean log-mass、下位quantile、high-confidence window 一致率を生成。truth は held-out candidate loss の評価だけに使い、top-1 regret、pairwise ranking accuracy、soft blend RMSEを比較する。
- **Full validation**: feature生成元も calibrator も whole-well outer fold を守る nested 5-fold。coverageがないwindowは明示的 missing とし、heatmap scoreなし、等重み、既存 path score、ML anchor、anchor-shrunk soft blendを比較。distance bucket・worst well・fold varianceを報告する。
- **Kill criterion**: real-GR score が shuffled-GR scoreより held-out candidate ranking / regretで優位にならない、selector regretが単純priorから15%以上縮まらない、または最終 blend が anchor を0.10以上改善しないなら終了。worst bucketが明確に悪化する場合も終了。
- **Coverage / selectability**: この案では existing five の coverage は固定。oracle union は性能根拠に使わず、主判定は **truth-free selectability**。heatmap candidatesを追加する版はI03/I10として別に評価する。
- **Hidden runtime / inference**: 保存済みheatmap生成が inferenceで再現可能であることを前提とし、logit samplingは O(KN), K=候補数。windowを全保持せずstream集約し、missing coverageに決定的fallback。offline、動的well数・可変suffix長、sample ID一対一、9時間内のend-to-end smokeを必須にする。
- **Family / confidence**: information / selector / **B**。

### I03 — Backbone-anchored heatmap lattice decoder

- **Mechanism / hypothesis**: 各rowのnodeを existing five paths の値と、十分なmarginを持つheatmap modesだけで構成する。通常はexisting backbone上を進み、heatmap branchへの遷移は値・局所slope・prefix boundaryが連続する箇所に限定し、短いbranch後はbackboneへ戻す。unaryはheatmap logitと既存path score、pairwiseは滑らかさとswitch penalty。strong backboneを捨てず局所的にnew-best modeを拾えば、E04のunion headroomの一部をdeployableなsingle pathへ変えられる。
- **根拠 / 反証**: E04 の union oracle 2.7455、new-best 0.2525（1000-ft-plusで0.3174）が強い支持。E05 の stitched-only 46–50、full-tail-only 32.3331、weighted point 59.2721 と ranking weakness は、自由なheatmap stitchingに強く反対。
- **最も近い失敗との差**: E05 は learned pathsを自立したfull-well trajectoryへstitch/fillした。I03ではexisting pathが必ずbackboneで、heatmapは高信頼かつ短区間のbranch proposalに限定する。右端 extrapolationは禁止し、branchがないrowは既存path nodeのみ。whole-path selectionでもrowwise unconstrained gateでもない。
- **Cheap test**: (a) truth-awareだが遷移制約付きのlattice oracleを計算し、自由unionではなくこの表現の上限を測る、(b) 固定・非学習のlogit unaryで1-fold decodeし、oracle gainがrankingに残るか確認する。oracleとtarget-freeを別表にする。
- **Full validation**: transition penalty、branch長上限、margin閾値はtrain wellsのみで選び、outer whole-well 5-foldでdecode。最終的にはML anchorへOOFで決めたscalar shrinkを許し、anchor、existing-only lattice、heatmap-only stitch、union oracleと比較する。
- **Kill criterion**: constrained oracleが7.0 RMSEを下回らず自由union headroomの大半が遷移不能、またはtarget-free decodeがexisting-only decoderを0.20も改善しない、またはanchor-shrunk最終予測がanchorを0.10改善しないなら終了。switch集中によるworst-well破綻もkill。
- **Coverage / selectability**: まず constrained lattice coverage/oracleを測り、その後同じgraphでtruth-free decoderを測る。oracle改善だけでは継続しない。coverage 0 のrowでは既存backboneのみなのでfull-grid生成を保証する。
- **Hidden runtime / inference**: node数を `5 + capped heatmap modes`、transitionを局所疎graphに固定し O(NK²) のKを小さく保つ。well単位stream処理、deterministic tie-break、peak memory上限、可変row数。public-specific branchなし、formation columnsなし、sample ID整列、end-to-end 9時間未満。
- **Family / confidence**: candidate generation + structured representation / **C**。

### I04 — Full-suffix discriminative GR–typewell path energy

- **Mechanism / hypothesis**: candidate pathごとに、pathが示すdepthに沿ってtypewell GRを参照し、horizontal suffix GRとのmulti-scale normalized difference、cross-correlation、run-length一致、局所順位などをwell全体で集約する。真のcandidate lossが小さいpathを、他のexisting pathsをhard negativeとしてpairwise/listwise学習する。local heatmapやgenerative likelihoodより長距離sequence evidenceが候補rankを安定化する可能性がある。
- **根拠 / 反証**: E03 はGRにgenuine signal、E02のHMM/PFはtypewell likelihoodに一定の情報、E04は候補集合にheadroomがあることを示す。対してHMM/likelihood-PF blend 10.2697はanchor 9.5264より弱く、local heatmapのworst-well zero coverageはGRだけで決まらないwellの存在を示す。
- **最も近い失敗との差**: E02/E06 はrowwise generative likelihoodをstate searchへ入れた。I04は探索をせず、有限candidateを **sequence-level discriminative energy** で比較する。E03/E05のlocal point生成とも異なり、full suffixのaggregate evidenceとcandidate hard negativesに直接最適化する。
- **Cheap test**: existing five OOF pathsについて、低コストな3–5個のsequence scoreを計算し、truth-best candidateとのpairwise AUC、top-1 regret、well長/距離bucket別の安定性を測る。prefix truthはscoreに使わず、必要ならGR amplitude normalizationのfitだけをrevealed prefixで行う。
- **Full validation**: scorerはouter foldのtrain wellsのみで学習。held-out wellではcandidate生成、feature生成、weightingをすべてtarget-freeにし、hard top-1とsoft posteriorを別評価。ML anchorへのshrink blendまでを最終比較とする。
- **Kill criterion**: simple HMM/path scoreを越えるcandidate regret改善が15%未満、worst-distance bucketでrankingが逆転、または最終soft predictionがanchorを0.10改善しないなら終了。typewell参照をcandidate TVTへ一意に写せない場合はmechanism自体を終了。
- **Coverage / selectability**: existing fiveのcoverageは固定し、評価対象はselectability。candidateを増やす場合は追加前後のoracle coverageと、固定scorerによるtarget-free RMSEを別々に報告する。
- **Hidden runtime / inference**: K本のpath上だけをscanするO(KN)設計とし、全depth gridやparticlesを展開しない。prefixから得るnormalizationはrevealed values/GRだけ。offline、well単位chunk、deterministic resampling、可変typewell長・suffix長、sample alignment、9時間wallを満たす。
- **Family / confidence**: information / sequence selector / **B**。

### I05 — Anchor-shrunk posterior mean over experts

- **Mechanism / hypothesis**: ML anchor、existing five、HMM/PF、利用可能ならheatmap-conditioned proposalをexpertとし、uncertainty/disagreementからOOFでcandidate loss posteriorを推定する。RMSEに対してhard top-1でなくposterior meanを出し、posteriorが曖昧ならweightをML anchorへ縮める。誤選択のtail riskを抑えながらoracle diversityの一部を得る。
- **根拠 / 反証**: E01のstrong anchor、E02/E04のlarge oracle gap、E05のrank inversionが支持。E06のhard switch/dense gate failureとworst regressionは、過信したdiscrete selectionに反対する。candidate residualの相関はpacketでは不明。
- **最も近い失敗との差**: E06のhard switchはsignalで1候補を選ぶ。I05はanchorを明示的priorにし、weight entropy floor、最大非anchor weight、well-level risk calibrationを持つsoft posterior mean。same-well prefix GRだけでgateしない。
- **Cheap test**: OOF predictionsと既存uncertaintyだけで、nonnegative ridge/temperature-scaled softmax/anchor-onlyの3つに限定したcross-fit blendを比較。oracle bestとのgap、weight calibration、worst-20 wellsを測る。
- **Full validation**: blend featureとweight learnerをnested whole-well foldsでfitし、row-pooled MSEに加えworst-well guardrailをtrain criterionにする。I02/I04のscoreを足す版は、それぞれのbase scoreがouter-fold-safeな場合だけ追加する。
- **Kill criterion**: equal/static blendよりheld-out gainが0.05未満、anchor比0.10未満、非anchor weightがfold間で不安定、またはworst-20-well RMSEが5%以上悪化なら終了。
- **Coverage / selectability**: union coverageはexpert集合のpropertyとして別集計。I05はsoft selectability/fusionだけを評価し、oracle best weightをfeatureやtraining target以外の推論に使わない。
- **Hidden runtime / inference**: base predictions生成後はO(KN)、固定小K、vectorized、deterministic。expert missing時のrenormalizationを事前定義し、hidden cardinality・public IDに依存せずsample IDへ整列。base全体を含むend-to-end 9時間を判定する。
- **Family / confidence**: fusion + uncertainty / **B**。

### I06 — Candidate-conditioned heatmap residual features for the ML anchor

- **Mechanism / hypothesis**: 各rowで、anchor値およびexisting path値の近傍にあるheatmap logit mass、modeまでのsigned distance、entropy、top-1/top-2 margin、候補間で支持が割れる方向を特徴化し、anchor residualを予測する。heatmapをtrajectoryにdecodeせず、MLが既に強い場所ではゼロ補正を学ぶ。
- **根拠 / 反証**: E01のstrong anchorとpersistent misses、E03のreal GR signal、E04のheatmap追加oracle gainが支持。E03のdirect output weakness、E05のpoor point pathsは大きな補正に反対する。
- **最も近い失敗との差**: E03/E05はheatmapからpoint pathを構成。I06はanchorを置換せず、candidate座標に条件付けた **evidence feature** として使う。E06のdense gateと異なりexpert切替ではなく連続的でclipされたresidual correction。
- **Cheap test**: frozen OOF anchorに対し、保存済み5-fold heatmapから5–10個のsummaryだけを作り、train-fold ridgeでheld-out residualを予測。shuffled-GR heatmap summaryをnegative controlにする。
- **Full validation**: deterministic ML pipelineに同じOOF-safe summaryを追加してwhole-well 5-fold再学習。base anchor、real-GR feature、shuffled-GR feature、entropy-onlyを比較し、distance/worst-well/fold varianceを確認する。
- **Kill criterion**: real-GR featureがshuffled controlを明確に越えない、frozen cheap testで0.05未満、fullでanchor比0.10未満、またはclip境界に補正が集中するなら終了。
- **Coverage / selectability**: heatmap coverage外はmissing flag + zero evidence。candidate setのoracle coverageは増えない。hard selectabilityを解かず、conditional residualとして情報利用するため別物として記録する。
- **Hidden runtime / inference**: heatmap inferenceが必要。summaryをrow単位stream生成しraw logits全保持を避ける。train-only formation columns不使用、可変well/window数、deterministic missing fallback、sample ID整列、全base込み9時間内。
- **Family / confidence**: information / feature role change / **B**。

### I07 — Persistent-error corruption curriculum for candidate scoring

- **Mechanism / hypothesis**: train truth pathまたは良好candidateを基準に、prefix boundaryを保つoffset drift、slow slope drift、単一regime jump、tail-only bendを注入してhard negative pathsを作る。score modelに「局所GRが似ていても長いpersistent missは高cost」と学習させる。実候補5本だけでは少ないfailure geometryを補える可能性がある。
- **根拠 / 反証**: E01のlong correlated errors、E05の大きなrank inversionが支持。実候補の誤差生成過程がこのcorruption familyに一致する証拠はpacketになく、synthetic shortcutの危険が大きい。
- **最も近い失敗との差**: closestはE05のpoor candidate calibrationだが、packet内にsynthetic hard-negative trainingの記録はない。新規性はモデルbrandでなくcandidate-error distributionのdata generationにある。
- **Cheap test**: train wellsの一部で4 corruptionを作り、simple sequence scoreを学習。評価はsyntheticではなくheld-out **real existing five** のpairwise ranking/regretだけで行い、corruption typeをfeatureから識別できるartifactは除く。
- **Full validation**: corruption amplitudeはtrain-foldのreal candidate residual distributionだけから決め、outer whole-well foldsでI04型scorerまたは小型rankerを学習。no-corruption版と比較し、fold/距離/worst-wellを確認する。
- **Kill criterion**: synthetic上だけ改善してreal-candidate regretが10%以上縮まらない、corruption sourceを当てるshortcutが見つかる、または最終anchor-shrunk RMSEが0.10改善しないなら終了。
- **Coverage / selectability**: synthetic pathsはtraining negativesで、inference coverageには数えない。主目的はreal候補のselectability。coverage metricとranker metricを分離する。
- **Hidden runtime / inference**: corruption生成はtraining-only。hiddenでは固定scorerをK pathへO(KN)適用するだけ。offline、deterministic model/SHA、formation列不使用、可変well長、sample alignment、9時間内。
- **Family / confidence**: data generation / **C**。

### I08 — Episodic prefix-to-suffix latent adapter

- **Mechanism / hypothesis**: train wellを複数cutoff episodeにし、revealed prefixで観測したanchor/candidate residual、boundary slope、GR alignment、uncertaintyから2–4次元のwell stateを推定する。そのstateからsuffix全体の低周波residual basis係数を出す。単純linear extrapolationでなく、train wellsで「prefix failure patternからsuffix driftへ」のmappingをmeta-learnする。
- **根拠 / 反証**: E01のpersistent miss、prefix TVTが推論時既知、E06のweak prefix observationが全distance bucketを改善したことが支持。E06のwellごとの符号混在とworst regression、mask distribution不明が反証。
- **最も近い失敗との差**: E06はprefix GRをlikelihoodへ足した。I08はrevealed prefix TVT residualでlatent error stateを推定し、suffix correctionを低rankに制限する。I01との差は固定offset/slope extrapolationでなく、多数のtrain-well episodesから非同一なprefix→suffix mappingを学ぶ点。
- **Cheap test**: OOF anchor residual matrixをsuffix normalized coordinateへ揃え、train foldsでSVDし上位2–4 basisを得る。prefix summaryからheld-out suffix basis係数をridge予測し、oracle basis headroomとtarget-free predictionを分けて測る。
- **Full validation**: all pseudo-cutoffs of one wellを必ず同じouter foldへ置き、cutoff length bucketsを本番想定に合わせる。adapter dimension、clip、anchor shrinkはinner foldsのみ。overall/long suffix/worst wellsでanchorとI01を比較する。
- **Kill criterion**: low-rank oracleがanchorを0.5以上改善できない、またはtarget-free adapterがI01を0.05も越えない、またはheld-out anchor比0.10未満なら終了。cutoff availabilityを再現できなければ停止。
- **Coverage / selectability**: candidate union coverageを増やさず、well-state correctionを行う。candidate summaryを入力にしてもtruth-best labelは推論に使わず、selector性能とは別に評価する。
- **Hidden runtime / inference**: fixed small adapterとbasis評価はO(N)。revealed prefixだけからstateをfitしsuffix truthは不使用。public cutoff長を固定せず、短prefix時fallbackを事前定義。offline/deterministic/sample alignment、追加memory小、9時間内。
- **Family / confidence**: representation + test-time adaptation / **C**。

### I09 — Risk-controlled segment expert switching with hysteresis

- **Mechanism / hypothesis**: suffixをcandidate disagreement、heatmap margin変化、trajectory changeで少数segmentに切り、segment単位でML anchor/existing pathsを選ぶ。transition penaltyとminimum segment lengthを持ち、不確実なsegmentはanchorへrejectする。whole-well top-1より細かく、rowwise gateより安定なscaleでtail-specific new-best rateを利用する。
- **根拠 / 反証**: E04のoverall/1000-ft-plus new-best差、E01のpersistent error、E05のrank inversionが支持。E06のdense gate/hard switch failureとE05のstitch failureは自由な局所切替へ反対。
- **最も近い失敗との差**: dense gateはrowwiseまたは高密度にsignalで切替えたと解釈される。I09はtruth-free change points、minimum duration、hysteresis、anchor rejectを持つ **block decision**。I03のvalue-level latticeと違い、既存expertのsegmentをそのまま保つ。
- **Cheap test**: 事前固定3–8 segment上限でtruth-aware segment oracleを計算し、whole-well oracleとの差とswitch数を測る。次にcached uncertaintyだけのsimple multinomial gateをcross-fitする。
- **Full validation**: segmentationとgateはouter fold train wellsのみでfitし、held-out wellで固定。anchor-only、whole-well selector、rowwise gate、segment gateを比較し、switch境界のerror spikeとworst-wellを監査する。
- **Kill criterion**: constrained segment oracleがwhole-well selectorより0.3も改善しない、target-free gateがwhole-well soft blendより0.10改善しない、またはswitch近傍でlarge missが増えるなら終了。
- **Coverage / selectability**: segment oracleはcoverage/representation上限、learned gateはselectabilityとして別報告。coverageのある候補だけでsegmentを構成し、missing時はanchor。
- **Hidden runtime / inference**: segment数を固定上限、DPはO(NSK²)の小K/S。well単位、deterministic tie-break、可変suffix長、offline、sample alignment。base predictions込み9時間内。
- **Family / confidence**: structured fusion / **C**。

### I10 — Heatmap-guided low-frequency warps of strong backbones

- **Mechanism / hypothesis**: heatmap modeを直接stitchせず、existing five pathの各々に、prefix boundaryを固定した1–2 knotの滑らかなwarpを数本だけ施し、高margin heatmap modeの集団へ近づける。candidateは常にstrong backboneの近傍にあり、局所ノイズやright-end extrapolationを避けながらunion diversityを増やす。
- **根拠 / 反証**: E04でheatmap candidates追加がoracle 5.0687→2.7455、long suffixでnew-best率0.3174。E05ではheatmap-only pathsが32–50 RMSEと弱いがexisting unionへは改善を加えるため、backbone近傍proposalの余地がある。
- **最も近い失敗との差**: E05はlocal pathsをfull-wellへstitch/fillし、57%近いright-end extrapolationも生じた。I10はexisting pathを全rowのbaseとし、low-frequency bounded warpだけを候補化し、heatmap modeの途切れを補間根拠にしない。
- **Cheap test**: 各backbone×高々4 warpをcached dataで生成し、まずfull-grid oracle coverage/RMSE、新-best率、候補間距離を測る。次にI02の固定scoreを新候補へそのまま適用し、coverage gainがselectabilityへ移るかを見る。
- **Full validation**: knot数、warp clip、proposal数はtrain foldsのみ。whole-well 5-foldでcandidate-generation oracleと、独立OOF scorerによるfinal predictionを別評価。candidate数増によるmultiple-choice過学習も監査する。
- **Kill criterion**: oracle unionがexisting fiveから0.5未満しか改善しない、候補数を増やすほどtarget-free regretが悪化、またはfinal anchor-shrunk予測が0.10改善しないなら終了。
- **Coverage / selectability**: 第一段階はcoverage/diversity、第二段階は固定scorerでselectability。oracleのみ改善してselectorが悪化する場合はdeployable案としてkillする。
- **Hidden runtime / inference**: proposal数をwell当たり固定上限（例20、値はinner CVで削減）にし、warp生成O(KN)。heatmap missing時はproposalなし。右端extrapolationなし、deterministic、stream処理、dynamic IDs、sample alignment、全体9時間内。
- **Family / confidence**: candidate generation / **C**。

### I11 — Deterministic O(KN) observation cache and path-only scorer

- **Mechanism / hypothesis**: horizontal/typewell GRのnormalization・multi-scale summaries・interpolation indexをwellごとに一度だけ作り、candidate path上の値だけをbatched gatherしてI02/I04/I09/I10のscoreを計算する。particlesやfull state gridを再探索しないことで、候補選択のiterationとhidden-safe inferenceを9時間内に可能にする。
- **根拠 / 反証**: E06のtested CPU pathは10h50m、E07は14.88 CPU hoursでwall超過。既存candidate pathsは再利用可能。反証は、full likelihoodをpath-only featuresへ落としたときranking情報が失われる可能性と、heatmap生成自体のruntimeがpacketにないこと。
- **最も近い失敗との差**: E06はsame-well prefix termをHMMのstate computationへ追加、E07はparticle smoothing。I11は新しいstate searchをせず、有限K pathのobservation featureを一括評価するcompute regime変更。単なるコード高速化ではなく、候補数・sequence score・nested validationを探索可能にするenabler。
- **Cheap test**: 代表的な長さのheld-out wells（選定はtrain-foldの長さquantileのみ）で、(a) score numerical parity、(b) rows/sec、(c) peak RSS、(d) repeated-run hashを測る。全773 OOF cache生成時間を外挿する。
- **Full validation**: 773 wellsをdeterministic shardsでcacheし、I02またはI04の一つをfull OOF実行。hidden約200 wellsはrow数/longest-wellベースの保守的projectionに加え、runtime sampleでend-to-end smokeを行う。
- **Kill criterion**: scorerに必要な統計を保持できない、同じ入力でhash不一致、path score部が全wallの20%以上を残してiterationを解禁しない、またはhidden projectionでend-to-end 7時間以内（9時間に対する暫定2時間buffer）を満たさないならcompute-enablerとして終了。
- **Coverage / selectability**: cacheはcoverageを増やさない。selectability featureを高速に供給するだけで、accuracyはI02/I04のheld-out評価で別判定する。
- **Hidden runtime / inference**: O(KN)、固定小K、well単位stream、preallocated arrays、bounded threads、offline。cache keyにmodel/feature SHA・fold・normalization versionを含め、hiddenではpublic SHAやwell listを仮定しない。runtime sample IDsを正として出力を再整列。
- **Family / confidence**: compute enabler / **B**。

### I12 — Multi-cutoff, well-blocked selectability training and stress test

- **Mechanism / hypothesis**: 1 wellから複数の擬似cutoff episodeを作り、candidate selector / prefix adapterをsuffix length・distance・availabilityの異なるregimeで学習する。ただし同一wellの全episodeを同じouter foldに閉じ込める。候補選択が特定cutoffに過適合しているなら、deployment variationを再現すること自体がgeneralizationを改善する。
- **根拠 / 反証**: hiddenはpublic 3 wellsと異なり約200 wells、suffix distance bucketでsignal差があり（E04/E06）、trusted validationはwhole-well GroupKFold。反証は、本番cutoff分布がpacketでは不明で、人工cutoffがdeploymentを歪める可能性。
- **最も近い失敗との差**: packetのtrusted GroupKFoldはwell leakageを防ぐが、複数availability regimeを学習・stress testした記録はない。I12はfoldを変えずに、outer group内でのみmask distributionを拡張するdata/validation mechanism。
- **Cheap test**: 既存OOF assetsで各well 2–3 cutoffsを再構成できるか確認し、同じcandidate scoreのranking/regretがcutoff間で安定するか測る。episodeがouter foldを跨がないassertionを自動検査する。
- **Full validation**: outer held-out wellの本番相当cutoffを最終評価専用にし、train wells内のmulti-cutoffでI01/I02/I08の一つをfit。short/long prefix、short/1000-ft-plus suffix、coverage missing、worst wellsを別集計する。
- **Kill criterion**: 本番相当maskを再現できない、multi-cutoff学習がsingle-cutoff版を0.05も改善しない、mask bucket間で符号が不安定、または同一well leakageを完全排除できないなら終了。
- **Coverage / selectability**: cutoffごとにcandidate coverage/oracleとtarget-free selector regretを別々に保存する。episode増加をcoverage改善と数えず、held-out unseen wellだけでselectabilityを判定する。
- **Hidden runtime / inference**: multi-cutoffはtraining/validationのみ。hiddenでは実際のrevealed prefix/suffixを1回処理する。短prefix・missing candidateのfallbackを固定し、dynamic wells/sample IDs、offline、determinism、9時間内を維持。
- **Family / confidence**: data generation + validation / **C**。

## Top 5 portfolio

| Priority | Idea / slot | なぜ portfolio に入れるか | Staged decision |
|---:|---|---|---|
| 1 | **I02 / safe** | 保存済みlogitsとexisting OOF pathsだけでcheap testでき、E03のgenuine GR signalをE05で失敗したpoint-path roleから切り離す。selector bottleneckを最短で直接検証する。 | cached path-score proxy → nested full OOF soft selector → hidden-sized inference smoke。ranking regret 15%改善が最初のgate。 |
| 2 | **I01 / safe** | E01のpersistent missへ直接作用し、revealed prefix TVTという合法なper-well supervisionを低自由度で使う。計算・実装リスクが最小。 | offset-only replay → bounded low-rank whole-well OOF → dynamic-well CPU smoke。worst-well guardrailを先に固定。 |
| 3 | **I03 / exploration** | E04の最大headroomを狙う非連続なrepresentation change。E05の失敗をstrong-backbone constraintで正面から回避する。 | constrained oracle graph → fixed-logit target-free decode → nested 5-fold + anchor shrink → capped-node smoke。oracleとselector両方をgate。 |
| 4 | **I04 / orthogonal** | local heatmapとは異なるfull-sequence evidenceでcandidate selectabilityを狙い、GR signalの距離方向の蓄積を使う。I02と誤差源が異なる可能性がある。 | 3–5 scalar sequence scores → hard-negative listwise OOF → soft posterior + anchor → O(KN) smoke。 |
| 5 | **I11 / compute_enabler** | E06/E07で既にwall超過が顕在化しており、I02/I04/I03の反復をhidden-safeにする共通基盤。accuracy案と独立のruntime kill条件を持つ。 | numerical parity + rows/sec → all-OOF deterministic cache → one selector full run → 200-well/row-count conservative projection。 |

Portfolio は `information/selector`、`representation/prefix adaptation`、`structured candidate generation`、`sequence evidence`、`compute` の5系統（少なくとも4 mechanism families）を含む。I02とI04はどちらもselectorだが、局所discriminative heatmap evidenceとfull-suffix typewell sequence energyでobservation scaleが異なる。I11は単独のscore改善案ではなく、9時間contractを満たさない案を早期に落とし、探索を実行可能にするslotである。

## Adversarial gate と reject した近接案

| Rejectした案 | failed gate | reopen condition |
|---|---|---|
| heatmap probability-weighted full-tail point pathを再調整 | E05で59.2721、top-5-only 32.3331。closest failureとの差がparameterだけで、ranking/calibration failureを解かない。 | 独立したtarget-free rank calibrationが先に成立し、同じartifactでheld-out selector regretが大幅に縮む場合。 |
| larger heatmap window / geometry channelの追加だけ | tested larger windowはtop-3低下、geometryはtop-3低下。parameter/channel tuningだけでmechanism changeがない。 | 新channelがinference-availableで、shuffled/no-signal controlに対し独立したcoverageまたはselector evidenceを示す場合。 |
| prefix-GR likelihoodのweight強化 | E06でstronger weightがgainをほぼ消し、worst regressionと10h50m wall超過。 | weak bounded conditionerとしてI02/I04のOOF selectorに移し、I11相当のruntimeとworst-well guardrailを満たす場合。 |
| lag128/256 smootherを直ちにfull run | lag64は大幅悪化、14.88h、seed mismatchで因果未同定。高コストのまま次lagへ進む根拠が弱い。 | frozen-controlとseed-pairedの極小ablationでunsmoothed tail parityを確認し、短runで改善方向とwall内projectionを示す場合。 |
| truth oracleでcandidateを選ぶstacker | leakage / same-OOF selection。E04 oracleはdeployable selectorではない。 | outer whole-well OOFでtruthはtrain-foldのloss targetに限定し、held-outではtarget-free featuresだけを使う場合（I02/I04/I05）。 |

## 全案共通の validation / inference guard

- outer split は whole-well GroupKFold。selector、blend、calibration、threshold、warp、transition penalty、synthetic distributionはすべてouter held-out wellを見ずに決める。同一wellの複数cutoffは同一foldに固定する。
- truth-aware oracleは **representation/coverage診断だけ** に使用し、target-free score、regret、最終RMSEを別表にする。同じOOF上でのweight/threshold選択をしない。
- 主比較は reproducible ML anchor 9.5264。pooled RMSEだけでなく、distance bucket、1000-ft-plus suffix、worst wells、fold variance、large persistent missを監査する。
- inferenceはruntimeのcompetition rootと`sample_submission.csv`からwell/ID/row orderを動的に決め、欠損・重複・余分なIDを検査する。public 3 wells固有の分岐、train-only formation columns、online dependencyを禁止する。
- end-to-end で9時間未満。row数・longest-wellに対する保守的runtime projection、peak RSS、deterministic repeat hash、seed/model/cache SHAを記録する。単体scorerが速くてもbase prediction/heatmapを含む全体wallで判定する。

## Portfolio の実行順序（提案のみ）

1. I11 のmicro-benchmarkと同時に、I02 のcached cheap test、I01 のprefix availability checkを行う。
2. I02がranking gateを通ればfull OOFへ進め、通らなければI04のorthogonal sequence scoreへ切り替える。
3. I03はまずconstrained oracleだけでrepresentationを反証し、上限が残る場合のみtarget-free decoderへ進む。
4. 最終候補は必ず `cheap proxy → full whole-well OOF → hidden-sized inference smoke` の3段階を通し、各cardのkill criterionを途中で緩めない。

