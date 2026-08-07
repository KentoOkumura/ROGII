# exp209 HMM persistent offset 原因監査

## 結論

exp209 exact HMM の persistent offset は単一原因ではない。ただし、支配的な構造は
かなり明確になった。

1. HMM は長い区間で符号の変わらない、ほぼ一定の TVT offset basin に入る。
2. `0.35 ft`位置gridと狭いprocess noiseの解像度不整合により、実効`0.1225 ft`の
   position kernelがsub-grid変位の一次モーメントを0方向へ縮める。episode開始前128行の
   truth-centered累積kernel biasは最終offsetの符号と85.1%一致し、正しいrate stateから
   始める制御合成HMMでもsigma補正またはexact-mean transportがfiltered RMSEを大幅に
   削減した。したがって、この離散化はoffsetを生成できる条件付き因果機構である。
3. ただし、実wellのprefix rateからGRなしで伝播したtransition-only priorでは、
   current priorの符号がactual offsetと75.39%一致し、その群だけでepisode SSEの
   85.32%を占めた。prior error絶対値medianは`36.75 ft`、actual HMMは`13.92 ft`で、
   HMMは大半を修正するが履歴方向を完全には消せていない。そのうえで
   exact-mean化はRMSEを`39.78 → 95.24 ft`へ悪化させ、position補正効果と実offsetは
   Spearman `-0.4656`、符号一致`32.45%`だった。anchor-safe subsetでも
   Spearman `-0.1343`、符号一致`43.10%`である。現行のposition shrinkageは、
   staleまたはmisspecifiedなprefix rateに対するregularizerにもなっており、
   position kernel単独をactual offsetの無条件root causeとは呼べない。
4. `mom=0.998`の0向きrate mean-reversionも、truth-centered局所式と制御合成HMMでは
   driftを作れる。しかし実prefix transition-only priorで`momentum=1`にした効果は
   実offsetとSpearman `-0.2349`、符号一致`43.57%`で、これもactual rate posteriorを
   固定しない限り単独の方向seedとは確定できない。補償時間proxy median 156 /
   p90 506行、rate半減期約346行、実測offset onset中央値232行という時間尺度の一致は、
   sticky rate dynamicsが長い履歴を作れる証拠として維持する。
5. position transitionは絶対TVTでなく前後差だけを見るため、grid内部ではpath全体の
   平行移動に厳密に不変である。initial anchorは開始時にしか入らず途中の再anchorはない。
   一時的なrate誤差でdatum差を作った後にlocal rateだけが再同期すると、transitionには
   offsetを戻す力がなく、GRだけがabsolute datumを復元する。このcontinuous
   translation-gauge lockがpersistent parallel offsetの構造原因である。
6. truth pathのhard support違反率はpersistent / nonpersistentで
   `0.01424% / 0.01431%`と同等で、真値がHMM grammarから排除されることは全体原因ではない。
   一方、真値pathのexact two-step transition NLLはpersistentで`0.24785 / row`、
   nonpersistentで`0.23458 / row`、global Viterbiはpersistentで`0.10851 / row`だった。
   onset前の非重複ringではtruth NLLが256--512行前の`0.24656`から直前16行の
   `0.37585`へ上昇した。これは真の局所rate変化にsticky/coarse grammarが追従しにくくなり、
   lagからdatum差を形成するsoftな発火機構を支持する。ただしtruth-lateの二段局所診断で、
   episode RMSEとの相関は弱く、actual rate posteriorを固定した因果介入ではない。
   保存posterior meanの差分でこのtransition負荷をrateへ分解すると、真の`|rate|`は
   `0.03647 → 0.04318`へ増えた一方、decoded `|rate|`は`0.03517 → 0.03095`へ減り、
   `|decoded rate - true rate|`は`0.01581 → 0.03365`へ`2.28x`増えた。真のrate acceleration
   は`0.00850 → 0.00813`で増えておらず、一般的な急カーブではなく、持続的なrateへの
   under-responseがdatum差を積分する運動学的な形成機構である。
   最大のprior+truth-GR群では直前のmoving rowsの`73.08%`が真のrateと同方向だが
   絶対値の小さいzero-directed under-responseで、rate誤差量の`64.66%`を占めた。
   反対にprior-opposed+candidate-GR群では`93.45%`がsame-direction overshootで、
   誤差量の`96.52%`を占める。縮みとwrong-GR過走は別経路である。
   同じringのraw観測GRは全体平均では常にtruth側で、transition負荷が上昇したepisodeの
   `54.94%`、SSE加重`66.03%`は直前16行でもtruth側だった。したがって多くの重いepisodeでは
   wrong-GRが最初にmodeを選ぶのではなく、transition/rate lagが先行し、GR aliasは別群の
   initiatorまたは形成後のlock増幅器として働く。
   raw GR missing率も256--512行前`31.62%`から直前16行`30.40%`へ微減し、missing変化と
   rate-error増分のrhoは`-0.0275`だった。最大prior+truth-GR群も
   `33.67% → 32.38%`であるため、missing gapへの突入は一般的な発火条件ではない。
7. raw GR が別深度を実際に支持する GR match / depth alias 群も存在する。episode内の
   GR NLL差はlag-1相関0.883、IAT約23.9行で、同方向evidenceが繰り返し加算される。
8. candidateからtruthへdepth pathを連続補間した観測GR NLLは強く非凸である。終点で
   truthが良いepisodeだけに限定しても、途中のbarrierはmedian 61.82 NLLで、20 NLL超が
   SSE加重95.32%を占めた。平均offsetだけを戻すconstant datum shiftでも、終点が改善する
   subsetのbarrierはmedian 41.46 NLL、20 NLL超がSSE加重83.48%だった。したがって正しい
   datumが終点で有利でも、GR emissionはそこへ単調に戻す復元力ではない。
9. それ以上に、raw GR は真値側を支持しているのに、sum-product posterior の総質量と
   posterior mean が別offset basinに残る群が大きい。この最大群ではsmoothed posterior
   stdが発症256--512行前`4.07 ft`、直前16行`5.39 ft`、episode内`7.05 ft`へ広がる。
   反対にprior opposed + candidate-GR群はepisode内`1.46 ft`で、confident wrong basinと
   broad conflictの二regimeがある。
10. global Viterbi はpersistent-offset区間の多くを回復できるため、最尤の1本の物理path
    自体がposterior meanより良い場合は多い。ただしepisode平均errorの符号は77.43%で
    posterior meanと同じで、5 ft以内まで解消するのは21.47%に留まる。したがってViterbi
    headroomの大半も完全なmode脱出ではなく、同方向offsetの縮小である。exact joint top-5も
    episode平均datum spanのmedian`0 ft`、p90`0.0010 ft`で、best-of-5はtop-1から
    `0.0009 ft`しか改善しない。上位path rankは別mode IDではなくmicro-path差に消費される。
    また最尤joint path単体のposterior probabilityはwell中央値で約`10^-466`、top-5合計でも
    約`10^-465`にすぎず、単一path IDはsum-product massのcarrierではない。
    rate差分でもglobal Viterbiのabsolute errorは発症256--512行前`0.02151`から
    直前16行`0.03196`へ`1.65x`増え、transition NLL crescendoとのrhoは`0.8364`だった。
    したがってrate lagはposterior meanの平均化だけではなく、合法な最尤joint pathにも残る。
    ただし直前16行ではViterbiがmeanより良いepisodeが`49.84%`、SSE加重`66.34%`あり、
    sum-product mass / posterior meanは重い形成期の増幅器ではある。
    exp270のlatent rate-state診断ではglobal Viterbiの切替回数が全773 wellsで
    median`0` / p90`2`、`75.03%`のwellsでsuffix全体zero-switchだった。
    persistent episodesの`67.40%` / SSE`73.29%`もzero-switch well上にあるため、
    stableなmax-product rate-mode IDを保持するだけではoffsetを防げない。
    制御合成HMMでは
   backward smoothingは正しいemission下の
   遅れを修復したため、backward演算自体が単独root causeではない。実データではtransition
   prior、構造化されたGR evidence、sum-productのpath multiplicity、backward message、
   posterior mean readoutの組合せが残る中心候補である。
11. 一方でglobal Viterbiを全行へ直接採用するとRMSEは悪化する。原因はwell・区間ごとに
   異なり、単純なdecoder置換や「stable mode idを常に保持」で解決しない。
12. exp408でpersistent 450 wellsをcurrent exp209 HMMのまま再decodeし、predictive
   prior、filtered alpha、smoothed posterior、backward beta、rate mass、log-sum /
   max-product差をepisode rowごとに保存した。638 episodes / 807,710 rowsの排他的分類は、
   forward transition/prior hysteresisがepisodes`70.85%`・SSE`59.40%`、
   backward smoothing reversalが`13.48%`・`23.04%`、sum-product path
   multiplicityが`5.80%`・`9.04%`、state support不足が`2.82%`・`6.39%`、
   mixedが`7.05%`・`2.12%`だった。raw-GR alias / imputation aliasが排他的主因の
   episodeは0だった。

したがって、過去の「GR matchingで違うmodeに入った」という説明は一部の区間には正しいが、
全体の説明としては不十分である。より正確には、

> prefixから持ち越したrate priorとsticky transitionが真のrate変化へ追従せず、
> その累積変位誤差が平行移動不変なabsolute-position chainを別datumへ運ぶ。
> current-row GR emissionは通常その時点でbasinを作るほど強くなく、相関した過去・将来の
> GR evidence、backward smoothing、sum-product path multiplicityが既にできたbasinを
> 固定・増幅する。coarse-grid position shrinkageは正しいrate basinではoffsetを作れるが、
> actual exp209の誤ったrate priorに対してはむしろ暴走を抑える側へ働く。

これがexp408のactual messageで確定した主因の順序である。GR matching説は完全な誤りでは
ないが、`current-row GRがwrong modeへ直接ジャンプさせる`という単独原因説は棄却する。
GRは一部episodeのseedまたは、より多くのepisodeでhistory/future evidenceとして
wrong basinをlockする条件因子である。

## exp408 actual-message直接監査

`exp408_hmm_message_rate_basin_audit`のKaggle CPU version 3は、450 / 450 wells、
2,264,135 suffix rowsを`15,930.997 sec`、peak RSS`3.588 GB`で完走した。
exp270とのposterior mean最大差は`0.0 ft`、正規化最大誤差は
`5.338e-08`、truth / episode read before freezeは`0 / 0`で、11 technical gatesを
すべて通過した。version 1はraw horizontal ID契約、version 2はNumPy reduction順序の
technical errorで停止し、version 3ではexp270と同じfloat64の`position→rate`加算順へ
戻した。科学条件や閾値は変えていない。

| 排他的主因 | episodes | wells | rows | episode比 | SSE比 |
|---|---:|---:|---:|---:|---:|
| forward transition/prior hysteresis | 452 | 350 | 557,692 | 70.85% | 59.40% |
| backward smoothing reversal | 86 | 72 | 129,308 | 13.48% | 23.04% |
| sum-product path multiplicity | 37 | 35 | 60,641 | 5.80% | 9.04% |
| state support shortage | 18 | 18 | 23,113 | 2.82% | 6.39% |
| mixed | 45 | 37 | 36,956 | 7.05% | 2.12% |
| raw-GR / imputation alias | 0 | 0 | 0 | 0% | 0% |

排他分類以外の重複条件ではforwardがSSE`65.78%`、backwardが`23.04%`、
path multiplicityが`72.09%`を覆い、forwardとmultiplicityの重複は197 episodesだった。
したがってmultiplicityは独立rootというより最大の重複増幅器でもある。

row-levelではpredictive priorのtruth-vs-wrong-basin log-oddsが`-log(3)`未満の行が
`70.35%`、SSE`69.15%`で、filtered alphaもほぼ同じだった。一方、current emissionが
truth oddsを`-log(3)`以上悪化させた行は`0.253%`、SSE`0.924%`にすぎず、
episode-dominantは0だった。backward betaはtruth oddsを強く悪化させる行が
`67.59%`、SSE`66.97%`で、wrong basinを新たに作った86 episodesが排他的backward群に
一致した。

最も直接的なrate証拠は、backwardでtruth近傍rate massが回復しながらtruth近傍position
massが悪化する行が`43.33%`、SSE`38.33%`存在したことである。rateは再同期しても、
それ以前の累積変位誤差でabsolute datumのtranslation gaugeが別位置へ固定されている。
filtered rateのzero方向under-responseはrows`70.91%`、SSE`70.36%`で、511 episodesを
dominantに覆った。episode平均のcurrent transition変位誤差とposition offsetは
Spearman`0.5693`、SSE加重符号一致`90.22%`だった。

exact-mean transitionの変位誤差はactual offset方向へrows`70.59%`、SSE`76.92%`で、
current quantization biasの同方向寄与はrows`32.97%`、SSE`28.24%`に留まった。
episode単位でもquantization biasとoffsetはSpearman`-0.3892`、SSE加重符号一致
`20.85%`である。したがってcoarse-grid shrinkageは条件付きでoffsetを作れるが、
actual exp209では誤ったrate外挿を弱めるregularizerになっており、無条件root causeではない。

GRについては、observed rowの直接GR差がcandidate側strongなのは`0.549%`、truth側strongは
`0.925%`だった。candidate-strong episode群は全SSEの`33.42%`を占めるためGRを無関係とは
できないが、この群の排他的forward原因は群内SSE`74.31%`、backwardは`22.55%`だった。
truth-strong群でもforward / backward / multiplicity / supportが群内SSE
`45.22% / 26.58% / 14.95% / 10.85%`を説明する。GRは系列相関したevidenceとして
prior/history/future messageへ統合されるが、current-rowのpointwise GR差を原因へ
直結させてはいけない。

閾値感度でも、forward dominant 50%はeffect thresholdを`0.1`から`log(3)`へ変えて
500→469 episodes、SSE`67.58%→65.78%`と安定した。current-emission dominant 50%は
effect`0.1`で6 episodes / SSE`0.074%`、`log(1.5)`以上では0である。raw-GR aliasも
effect`0.1`で16 episodes / SSE`0.457%`、`log(1.5)`で1 episode / `0.034%`、
`log(2)`以上で0だった。結論は閾値選択のartifactではない。

## 対象と方法

- decoder: `exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`
- 保存候補:
  `exp270_exact_hmm_posterior_mode_candidate_audit_candidates.csv.gz`
- 対象: `3,783,989 rows / 773 wells`
- 比較decoder:
  - posterior mean
  - marginal MAP
  - global Viterbi
- persistent offset:
  `abs(posterior_mean - truth) > 10 ft` が128行以上連続
- emission attribution:
  exp209のtypewell GR、zero-fill prefix sigma、capped Gaussian emissionを再構成し、
  各行で真値TVTとposterior mean TVTのGR NLLを比較
- evidence class:
  観測済みraw GR行におけるepisode合計NLL差が`+5`以上ならcandidate側を強く支持、
  `-5`以下ならtruth側を強く支持、それ以外をnear-tie
- truthはcandidate、feature、predictionの生成には使わず、診断のlate joinに限定

## Persistent offset の寄与

| 項目 | 値 |
| --- | ---: |
| episodes | 638 |
| episode wells | 450 / 773 |
| episode rows | 807,710 |
| 全行に占めるepisode row比 | 21.3455% |
| 全SSEに占めるepisode SSE比 | **91.9880%** |
| episode誤差符号一貫性 median | 1.000 |
| episode誤差slope median | -0.000237 ft / row |
| 最後のabs error 5 ft以内からepisode開始まで | median 232 rows |
| 同onsetが16 rows以内 | 0 / 638 |
| episode後にabs error 5 ft以内へ未復帰 | 385 / 638 |

定義感度として、abs error閾値`5 / 10 / 15 / 20 ft`とminimum run
`64 / 128 / 256 rows`の全12通りを再計算した。10 ftではrun長64 / 128 / 256で
SSE占有率が`92.1109% / 91.9880% / 91.5261%`、15 ft×128でも`83.5424%`、
20 ft×128でも`73.7385%`だった。全12条件でepisode内誤差符号一貫性medianは`1.0`。
canonical条件は638 episodes / 807,710 rowsへ完全一致した。したがって
persistent parallel-offsetによるSSE支配は10 ft×128行という定義の産物ではない。

RMSEを支配しているのは散発的な大誤差ではなく、長いほぼ定数offsetである。
またmode changeは瞬時のjumpではない。多くは約200--250行かけて5 ft以内から10 ft超へ
rampし、その後長時間固定される。candidate-strong / truth-strongのonset中央値もそれぞれ
200 / 250.5 rowsであり、GR alias群でも単一行のhard switchではなく、transitionと
posterior massが徐々に別basinへ移る。

episode直前128行のerror slope絶対値medianは`0.025201 ft / row`で、同区間に増えた
abs errorはmedian `3.271 ft`だった。episode内へ入るとabs slope medianは
`0.002673 ft / row`へ**9.43倍**縮み、ほぼ平行になる。pre128 slopeの符号は最終offsetと
93.73%、SSE加重95.89%一致するが、これはabs error閾値を越えたepisodeを事後選択している
ため符号一致自体にはselection effectがある。独立に重要なのは、形成期のrate/error driftが
固定期より約1桁速いことである。HMMは一時的なrate追従ずれでdatum差を蓄積し、その後
local rateを再同期してもabsolute offsetを戻せない「ramp → parallel lock」を示す。

episode内のtrue U-rate medianとposterior-mean pathのU-rate medianはSpearman `0.9080`で
一致する。誤差slopeもほぼ0であるため、HMMは局所shape / rateを完全に失うのではなく、
概ね正しい傾斜の平行pathを誤ったabsolute datumで追跡している。

### Position transitionの平行移動対称性とdatum lock

exp209のposition transitionは、有限grid端を除けば

```text
K_p(p_t - p_{t-1} - (r_t * dMD - dZ))
```

だけで決まる。任意の一定offset `c`について`p_{t-1}`と`p_t`をともに`+c`しても差は変わらず、
transition確率は厳密に同じである。rate transitionもabsolute positionを見ない。absolute
TVTのinitial position priorはsuffix開始時に一度だけ入り、suffix途中にgeometry unary、
datum prior、reset、再anchorは存在しない。

最終episode平均offsetと同じ量だけpath全体をsuffix先頭からずらす仮想pathは、
`start_sig=0.75 ft`のinitial priorだけで追加NLLがmedian `172.12`、p90 `695.00`になる。
これは実posteriorのescape probabilityではないが、別datum pathがsuffix先頭から同時に
立ち上がるより、後からrate差を積分して形成される必要があることを示す。実測pre128 error
slopeの絶対median `0.02520 ft/row`はrate grid `0.005`の約`5.04 cells`に相当し、
5 ftから10 ft閾値までの追加5 ftを作る時間は約`198 rows`である。観測onset median
`232 rows`と同じスケールで、prefix initial rateとpre128 true rateの絶対差median
`0.02`（4 cells）とも整合する。

したがって、形成期に一時的なrate mismatchで10--30 ftのdatum差が蓄積し、その後local
rateがtruthへ再同期すると、正しいpathとparallel-offset pathはtransition energyだけでは
区別できない。元へ戻るには、GR emissionが一時的なrate deviationと連続した中間positionの
transition costを上回り、posterior massを別datumへ移す必要がある。GRが周期的・相関的・
missingであればwrong datumはmetastableになる。これは「mode IDを失う」という離散的な
説明より、連続的なtranslation gaugeが誤ったdatumでlockする構造である。

有限grid端はこの対称性を破るが、保存posteriorのposition edge occupancyは0であり、
固定bandも全wellで十分だったため、actual episodeでは対称性を戻す方向には働いていない。
初期position priorの平均biasが`6.94e-17 ft`だった事実も矛盾しない。初期anchorは正確だが、
途中で再適用されないことが問題である。

この構造は、absolute geometry unaryのexp279がexp209より`1.9023 ft`改善したこと、
residual-offset HMMのexp281がpersistent episodesを`551 → 530`へ減らしたこととも整合する。
一方、exp279はtailを悪化させ、GR-only reset triggerのexp366はAUC`0.5000`・発火率
`0.00118%`だった。つまり欠けているのは単純な常時anchorや検出可能なhard resetではなく、
不確かなabsolute datum massをrate/historyとjointに再配分する機構である。

episode平均offsetの絶対値はmedian `13.92 ft`、p90 `27.96 ft`、最大`61.30 ft`で連続的に
分布する。負側は362 episodes / episode SSE 63.15%、正側も276 episodes / 36.85%ある。
単一の固定offset、1行indexずれ、TVT/Z符号反転のような一定量・単一方向の実装bugとは
整合しない。

### Truth pathのtransition grammar適合性

真値がHMMの状態・遷移grammarからhardに排除されてoffsetが始まる可能性を、全773 wells /
3,783,989 rowsで確認した。truth、posterior mean、row-wise marginal MAP、global Viterbiを
exp209の`0.35 ft`gridへ固定し、5点position supportと隣接3-state rate supportを順に監査した。
global Viterbiのhard違反は0で、実装と保存pathのparity guardを満たした。

truthのlocal illegal transitionはpersistent内`115 / 807,710 = 0.01424%`、
nonpersistent内`426 / 2,976,279 = 0.01431%`で、率はほぼ同じだった。onset前128行に
hard breakがあるepisodeは`2.3511%`、episode SSE内`3.4525%`だけである。persistent
episode全体では75 / 638 episodesに1回以上の違反があるが、違反rowはepisode rowsの
`0.01752%`にすぎない。隣接rate連続性を加えても追加breakは0だった。したがって
「真値pathが許容support外なのでHMMが別modeへ強制された」というhard exclusion仮説は、
全体原因として反証された。

ただしhard可否では見えないsoft transition costには差がある。前行の局所
position-conditioned rate分布をexact 3-state rate kernelで1 step伝播し、次の固定position
shiftを採点した二段NLLは、truthでpersistent`0.24785 / row`、nonpersistent
`0.23458 / row`、global Viterbiでpersistent`0.10851 / row`だった。episode平均ではtruthが
Viterbiより高コストなものが`99.2163%`、SSE加重`99.8923%`で、差とepisode RMSEの
Spearmanは`0.3037`だった。この比較の一部はViterbiがtransitionを最適化する定義上当然だが、
truth側の局所rate/position motionが現行grammarにとって相対的に高コストであることは示す。

さらに、onset前のnested windowをrow数で差分して非重複ringへ直すと、truth NLLは
256--512行前`0.24656`、128--256行前`0.28429`、64--128行前`0.33277`、
16--64行前`0.35094`、直前0--16行`0.37585`と、onsetへ近づくほど上昇した。全ringを
持つ619 episodesのうちnear NLLがfarより高いものは`67.2052%`、near-minus-far medianは
`0.11274`だった。near-minus-farとpre128 error slope絶対値のSpearmanは`0.4204`だが、
episode RMSEとは`0.1408`、Viterbi gainとは`-0.0866`に留まる。よってsoft grammar mismatchは
「局所rate変化へ追従できずlagを作る発火条件」を支持する一方、形成後のoffset量やdecoder
headroomを単独では説明しない。形成後はtruthのepisode NLL自体が`0.25939`へ下がるため、
一度できたdatum差がtranslation gaugeで平行lockする時間順序とも整合する。

同じ非重複ringで、補間値を除くraw観測GRだけをexp209 Gaussian emissionで再採点した。
正ならGRがposterior mean側、負ならtruth側を支持する`truth NLL - candidate NLL`は、
256--512行前`-0.00979`、128--256行前`-0.01037`、64--128行前`-0.01237`、
16--64行前`-0.02214`、直前0--16行`-0.02460 NLL / observed row`で、全体平均では
onsetへ近づくほどtruth側へ強まった。transition NLLのnear-minus-farとGR advantageの
near-minus-farはSpearman`-0.0552`、GR変化とpre128 error slope絶対値は`-0.0723`で、
transition負荷上昇とwrong-GR化は連動していない。

near/farの両ringにraw観測がある618 episodesのうち、transition負荷が上昇したのは415
episodes=`67.1521%`だった。その415 episodesの`54.9398%`、eligible SSEで条件付けると
`66.0339%`は、直前16行のraw GR合計もtruth側だった。最大の
`prior_persists_against_truth_gr`群では、この比率がepisodes`60.9589%` /
SSE`64.9977%`である。したがって最大群では「wrong GR matchへ入ったこと」が発火の説明に
ならず、truthを支持する観測に抗してrate/history側のlagが10 ftを越えた像が強い。

一方、`candidate_gr_with_opposed_prior`群では、transition負荷上昇episodeの`70.3704%` /
SSE`73.8243%`が直前16行でもcandidate側GRだった。この群ではwrong-GRが発火から関与する
別経路を維持する。全episodeを一つの原因へ潰さず、

1. transition/rate lagが先行し、truth GRに抗してdatum差を形成する経路
2. candidate側GR aliasが早い段階から別datumを支持する経路
3. 形成時はtruth側でも、その後の相関GR・非凸barrierがwrong datumを固定する経路

に分けるのが現時点の時系列証拠に合う。ただしposterior mean / Viterbi pathと
error-defined onsetは保存済みのsmoothed出力で、今回もtruth-late記述診断である。
forward alphaにおけるGR更新とtransition予測の先後を直接観測したものではない。

row-wise marginal MAPはpersistent rowsの`0.26742%`でhard違反し、638 episodesの
`53.9185%`、episode SSEの`87.5560%`に少なくとも1回の違反があった。したがって各rowの
最大posterior stateを並べた列は合法なjoint pathとは限らず、「marginal MAPをmode IDとして
保持する」解釈は不適切である。これはglobal Viterbiの0違反と対照的である。

この監査はGRを含まず、保存predictionへtruthをlate joinした固定trajectoryの局所採点である。
error-defined onset周辺を事後選択しており、long-history filtered rate posteriorも表さない。
したがってsoft costの時間上昇をactual alphaの因果効果へ確定するには、この時点では
message/rate posterior保存Stage Aが必要だった。後続exp408でこのgapは解消した。

## Raw GR evidence による分解

補間行を除き、raw GRが実際に観測された行だけでepisode evidenceを集計した。

| observed GR evidence | episodes | episode SSE内 | 全exp209 SSE内 | 解釈 |
| --- | ---: | ---: | ---: | --- |
| truth strong | 278 | 57.3536% | **52.7585%** | GRは真値側なのにHMMが別offsetへ残る |
| candidate strong | 180 | 33.4158% | **30.7385%** | GR match / depth aliasがwrong offsetを支持 |
| near tie | 180 | 9.2306% | 8.4910% | GR単独では識別困難 |

evidence classのNLL閾値を`±0 / 1 / 2 / 5 / 10 / 20 / 50 / 100`へ変えても、
全8条件でtruth-support SSEがcandidate-support SSEを上回った。episode SSE内の差は
最小`9.6656`、最大`25.5224` percentage pointsである。したがって「wrong-GR群は実在するが、
truth-GR群の方が大きい」という順序は`±5`の分類閾値に依存しない。

episode中央値でも分離は明確だった。

| evidence | raw-candidate GR corr | raw-truth GR corr | raw-candidate GR RMSE | raw-truth GR RMSE |
| --- | ---: | ---: | ---: | ---: |
| candidate strong | 0.6371 | 0.4368 | 9.078 | 13.740 |
| truth strong | 0.0737 | 0.6416 | 17.035 | 10.690 |
| near tie | 0.1689 | 0.5077 | 10.033 | 10.251 |

candidate-strong群では、実際にraw GRが真値より誤candidateのtypewell GRへよく一致する。
これはwrong-mode GR matchingが実在する直接証拠である。

ただしcandidate側とtruth側のtypewell GR同士のepisode相関はmedian `0.4222`だった。
説明用に相関`>=0.5`を「類似motif alias」と置くと77 / 180 episodesで、candidate-strong
SSEの37.36%、全exp209 SSEの約11.48%である。残りは同じtypewell内の明瞭な反復motif
というより、horizontal GRが対応truth区間のtypewell GRからずれ、別区間の方へ近くなる
reference mismatch / geological nonstationarityの成分が大きい。`0.5`は診断用の記述閾値で、
採用gateではない。

しかしtruth-strong群の方がSSE寄与は大きい。この群ではraw GRと真値側typewell GRの相関が
高く、誤candidate側との相関は低い。それでもposterior meanは長期offsetを維持するため、
row emissionだけを原因にはできない。

episode平均posterior stdはcandidate-strongで`2.318 ft`、truth-strongで`6.819 ft`だった。
前者はGRが誤modeを支持してposteriorも比較的集中する「confident wrong-mode」、後者は
真値evidenceとwrong-basinの履歴質量が競合する「broad competing-path」になっている。
同じoffset tailでも必要な対策は異なる。

### Candidate-to-truth GR landscape

保存済みposterior meanを固定し、各persistent episodeについて41点のGR emission NLLを
再構成した。次の2種類を分けた。

- pointwise truth morph:
  `candidate + f * (truth - candidate)`。終点ではdatumと局所shape/rate誤差をともに除く。
- constant datum shift:
  `candidate - f * mean(candidate - truth)`。candidateの局所shapeを保ったまま平均offsetだけを
  除く。

raw GR観測行のcandidate/truth端点NLLは既存ledgerと最大絶対差
`4.55e-13`で一致した。pointwise truth morphでは、truth終点がcandidateより良いのは
374 / 638 episodes、episode SSEの`60.9894%`だった。その改善subsetに限定しても、
candidateより悪くなる途中のbarrierはmedian `61.8245 NLL`、5 NLL超が
`91.1765%`のepisodes / SSE加重`97.2221%`、20 NLL超が`73.2620%` /
SSE加重`95.3217%`だった。最大の`prior aligned + truth GR`群216 episodesでは全件truth終点が
良いにもかかわらず、barrier median `74.2071 NLL`、20 NLL超が群内SSEの`95.3054%`である。

constant datum shiftはさらに厳しい。平均offsetを除いた終点で観測GR NLLが改善するのは
212 / 638 episodes、episode SSEの`18.1542%`だけだった。これはdepth RMSE上の平均biasを
除いても、candidateのrate/local phaseを固定したままではtruth側GR motifを再現できないことを
示す。その212 episodesに限定してもbarrier medianは`41.4616 NLL`で、20 NLL超は
`67.9245%` / SSE加重`83.4798%`だった。したがってpersistent pathは低次には平行でも、
HMM内の競合basinは単一のconstant mode IDではなく、absolute datumとrate/local phaseが
結合した非凸なpath familyである。

barrierとactual episode RMSEのSpearmanはpointwise / constantで`0.4570 / 0.4757`、
Viterbi gainとは`0.2003 / 0.1943`に留まる。GR landscapeの非凸性はmetastabilityを作る
強い要因だが、severityやdecoder headroomの単独決定因子ではない。またこの41点sliceは
truth-late診断であり、逐次HMMのminimum-action escape pathではない。正のbarrierはその
slice上の非凸性を示すが、actual predictive / alpha / beta massの遷移確率そのものではない。

## Decoder による分解

| decoder | 全行RMSE | persistent rows RMSE |
| --- | ---: | ---: |
| posterior mean | 11.9383 | 24.7831 |
| marginal MAP | 12.5925 | 25.9590 |
| global Viterbi | 15.5517 | **19.8422** |
| mean / MAP / Viterbi row-wise oracle | **7.5172** | **15.2329** |

global Viterbiがposterior meanより改善するepisodeは638件中386件だった。改善episodeを
大幅回復と部分回復に分けると、persistent episode SSEの82.8306%、全exp209 SSEの
**76.1943%**を占める。

| Viterbi outcome | episode SSE内 | 全exp209 SSE内 |
| --- | ---: | ---: |
| large recovery | 49.4408% | 45.4796% |
| partial recovery | 33.3898% | 30.7147% |
| not better | 17.1694% | 15.7938% |

これは、持続オフセットの大部分で「最良joint pathが完全に失われた」のではなく、
正しい、またはより良いcoherent pathが残っていることを示す。posterior meanを支配するのは、
少なくとも単一pathの最大scoreだけではない。ただし、この比較だけからwrong basinの
多数path、forward prior、backward messageの寄与を分離することはできない。

3 decoderのうち各行でtruthに最も近いものを選ぶ診断oracleは、persistent rowsのSSEを
posterior mean比`62.2%`削減する。これはtarget依存でdeployできないが、保存decoder候補の
範囲内にも大きい回復headroomがあり、「真値側state/pathが状態空間から完全消失した」だけでは
説明できないことを強める。observed GR evidence別のpooled値は次のとおり。

| evidence | posterior mean | marginal MAP | global Viterbi | row-wise oracle | oracle SSE削減 |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate strong | 25.0500 | 25.1771 | 18.6397 | 15.4000 | 62.21% |
| truth strong | 26.0440 | 27.9834 | 21.0453 | 15.7800 | 63.29% |
| near tie | 19.2132 | 19.7414 | 18.0531 | 12.7972 | 55.64% |

candidate-strong episodeではposterior meanとmarginal MAPの距離中央値が`0.498 ft`、
posterior std中央値`1.583 ft`で、marginal自体が比較的集中したconfident wrong-modeである。
truth-strongでは同距離`2.377 ft`、posterior std`5.739 ft`で、真値GRとhistory massが
競合するbroad posteriorである。この二型は同じ「mode offset」として一律処理できない。

large-Viterbi-recovery 205 episodesでは、posterior mean RMSE `27.4411`に対しViterbi
`8.0091`、row-wise oracle `7.8934`で、episode SSEの91.73%をoracleが除ける。一方、
not-better 252 episodesではoracle SSE削減は9.04%しかない。posterior stdとViterbi RMSE
gainのSpearmanは`0.0503`に留まり、単純なuncertainty thresholdでは回復可能群を
target-freeに識別できない。

ただし全行ではglobal Viterbiがposterior meanより3.6134 ft悪い。Viterbi回復可否を
target-freeに判定できないため、直接置換は解ではない。marginal MAPもpersistent rowsで
悪化しており、各行のmodeだけを保持しても不十分で、系列全体のpath identityが必要になる。

### top-K path が別basinを表しているか

exp270の保存済みpath diagnosticsを再集計すると、joint top-Kはmacroな別mode候補ではなく、
ほぼ同一pathの局所摂動で埋まっていた。

| 監査 | 値 |
| --- | ---: |
| unique top-2 pathが得られたwell | 655 / 773 |
| top-1 / top-2でrate系列が同一 | 652 / 655 |
| top-1 / top-2 TVT path間RMSE median | 0.006419 ft |
| 同p90 / p99 | 0.014151 / 0.350000 ft |
| 同row最大差 median / p99 | 0.35 / 0.35 ft |
| top-2 score gap median / p90 | 0.001099 / 0.007129 |
| top-1 rateがsuffix全体で一定 | 580 / 773 |

1グリッドが0.35 ftなので、典型的なtop-2は長い系列のごく一部を1グリッドずらしただけである。
top-2 score gapとposterior-mean well RMSEのSpearmanも`0.0176`で、tail severityを説明しない。

このwell全体診断を638 persistent episodes / 807,710 rowsへ限定して、exact joint top-5が
episode内でどのdatumを張るかも再監査した。top-5はtruthを見ずにjoint score順でfreezeされ、
full TVT pathが完全一致するものだけdeduplicateされている。

| persistent episode内top-5監査 | 値 |
| --- | ---: |
| 5 unique full-well pathsが利用可能 | 439 / 638 episodes、SSE加重77.56% |
| episode内top-K平均TVT span median / p90 | `0 / 0.001029 ft` |
| episode内pairwise path RMSE最大 median / p90 | `0 / 0.019545 ft` |
| episode内で利用可能top-Kが完全一致 | `70.69%` |
| top-1とのepisode平均差が1 ft超のalternativeあり | `1 / 638` |
| 同5 ft超 / 10 ft超 | `0 / 0` |
| rank-2 score gap median | `0.001038` |
| available rank最大score gap median | `0.002686` |
| truthをtop-K min/maxが挟むrow比 | `0.01164%` |
| top-Kのいずれかがtruth 5 ft以内のrow比 | `23.7257%` |

persistent row pooled RMSEはtop-1 `19.842201 ft`に対し、episodeごとにbest top-Kを選んでも
`19.841305 ft`、row-wise top-K oracleでも`19.841294 ft`で、追加4本による改善は約
`0.0009 ft`しかない。top-Kのどれかがepisode平均5 ft以内に入る比率
`21.4734%` / SSE`20.0658%`も、実質top-1 Viterbiだけの値から増えていない。
scoreはほぼ同点なのにpathもほぼ同一であり、上位rankは別datum basinではなく、数行だけ
1 grid変えたmicro-path degeneracyに消費されている。

exp270が保存した`path_log_posterior = joint path score - log likelihood`から、全773 wellsで
individual path massも直接確認した。最尤joint pathのposterior probabilityはlog10中央値
`-466.20`、top-5合計でも`-465.51`だった。top-5 massのtop-1比log gain中央値は
`1.6076 ≈ log(5)`で、5本はほぼ同scoreだが、全posterior massに対しては合計しても
天文学的に小さい。したがって長いMarkov系列で単一のglobal path IDを「現在のmode mass」と
みなすこと自体が不適切である。

一方、長さで割ったtop-1 surprisalはpersistent wellsでmedian`0.21851 nats/row`、
nonpersistent wellsで`0.22176`と、persistent側の方がむしろわずかに小さい。
persistent 450 wells内ではsurprisal rateとposterior-mean RMSEのSpearmanは`0.0047`、
persistent row率とは`-0.0246`だった。よって「posteriorが全般にpath-diffuseだからoffsetが
大きい」という総量仮説もseverity原因として反証される。必要なのは全path entropyではなく、
truth datum basinとwrong datum basinへ属する膨大なmicro-path群の**相対総質量**である。

したがって、exp270のtop-5に別offset basinが現れないことは「別basinがない」という反証では
ない。局所的な近傍pathの組合せ数がrankを消費しており、macro basin単位の総質量をtop-K列挙
では測れていない。同時に、exp270 top-K rankをmode IDとして保持する案には実際の別datum
候補がなく、回復機構にならないことは強く確定した。次診断ではtop-K pathを増やすのではなく、
truth / candidate / Viterbi近傍をbasinとしてまとめたforward・filtered・smoothed massを
直接保存する必要がある。

### Offset onset前後のposterior geometry

保存済みexp209 smoothed position posteriorのstd、row marginal mode mass / gap、
posterior meanとmarginal MAP / global Viterbiの距離を、transition/GR監査と同じ
non-overlapping pre-onset ringで再集計した。

全体ではposterior stdのepisode平均が256--512行前`3.1616 ft`、直前16行`3.9575 ft`、
episode内`4.8353 ft`へ増えた。far/near双方を持つ619 episodesの`71.2439%`、
SSE加重`78.1202%`で直前stdが増加した。同時にmarginal mode massは
`0.2667 → 0.2516`、mode gapは`0.07969 → 0.07273`へ低下し、mean--MAP距離は
`1.6242 → 2.0747 ft`へ拡大した。したがって多くのepisodeは、突然一つのhard modeへ
切り替わるだけでなく、発症前からsmoothed position massが広がりながらdatum差を形成する。

ただしcause群で形は大きく異なる。

| cause群 | std 256--512行前 | std 直前16行 | episode内std | mean--MAP episode内 |
| --- | ---: | ---: | ---: | ---: |
| prior opposed + candidate GR | `1.1914` | `1.1579` | `1.4609` | `0.4492 ft` |
| prior aligned + candidate GR | `2.1517` | `2.4837` | `2.5959` | `1.0016 ft` |
| prior aligned + truth GR | `4.0654` | `5.3874` | `7.0471` | `4.1288 ft` |
| neither prior nor observed GR | `3.1693` | `4.3798` | `6.0252` | `2.8076 ft` |

最大の`prior_persists_against_truth_gr`群ではstd broadeningがeligible episodes
`79.2453%` / SSE`81.7982%`である。一方、`candidate_gr_with_opposed_prior`群は平均stdが
far`1.1914`からnear`1.1579 ft`で増えず、episode内も`1.4609 ft`に集中する。これは

- candidate側GRが早くから一致する群: 比較的confidentなwrong basin
- truth側GRとprior/historyが競合する群: 発症前から広がるconflicted basin

という二つのposterior regimeを、GR NLLとは独立な保存posterior形状でも再確認する。
単純な「全episodeで一度だけmode switch」という表現は前者には近いが、支配的な後者には
不十分である。

一方、std near-minus-farとepisode RMSEのSpearmanは`0.0919`、
transition NLL crescendoとは`0.0414`、pre128 error slope絶対値とは`0.1293`に留まる。
posterior broadeningは経路分類と時間的な競合のmarkerだが、offset severityやtransition
発火の単独原因ではない。またこの保存値だけでは、broadeningがforward alphaですでに
存在したか、future evidenceを使うbetaで加わったかを分離できなかった。後続exp408では
exclusive forward / backward SSE比を`59.40% / 23.04%`へ分離した。

### 発症前transition pressureのrate-lag分解

truth grammarのNLL crescendoが、真の急なrate変化そのものかHMM側の追従遅れかを分けるため、
同じnon-overlapping ringで、truth TVTの一階差分を`true rate`、凍結済みposterior meanの
一階差分を`decoded rate`として再集計した。定義上、
`decoded rate - true rate`はTVT prediction errorの一階差分と全行で厳密に一致し、
最大差は`0.0`だった。

| cause群 | true `|rate|` far→near | true acceleration far→near | decoded `|rate|` far→near | `|rate error|` far→near |
| --- | ---: | ---: | ---: | ---: |
| 全体 | `0.03647 → 0.04318` | `0.00850 → 0.00813` | `0.03517 → 0.03095` | `0.01581 → 0.03365` |
| prior aligned + truth GR | `0.03760 → 0.04723` | `0.00867 → 0.00793` | `0.03314 → 0.02434` | `0.01543 → 0.03388` |
| prior opposed + candidate GR | `0.03258 → 0.02933` | `0.00798 → 0.00750` | `0.04330 → 0.06129` | `0.02015 → 0.03293` |

far / near双方を持つ619 episodesでは、true `|rate|`が増えたのは`58.1583%`、
SSE加重`59.8800%`だが、true accelerationが増えたのは`47.8191%`、
SSE加重`45.8826%`に留まった。一方、absolute rate errorは`78.9984%`、
SSE加重`85.5844%`で増え、pooled near / far比は`2.2848x`だった。その増分は
truth transition NLL crescendoとSpearman `0.6812`、pre128 error slope絶対値と
`0.6463`、episode RMSEと`0.3291`で対応した。したがってtruth grammar crescendoの
中心は、普遍的な高曲率・hard jumpではなく、徐々に大きくなるrate lagである。

最大の`prior_persists_against_truth_gr`群では、真の`|rate|`が増えるepisodeが
`68.3962%`、SSE加重`82.3366%`であるのに、decoded `|rate|`は平均で逆に減少した。
rate-error増分とtransition crescendoのrhoは`0.7538`、pre128 slopeとは`0.6823`である。
raw GRがtruthを支持しているため、この経路は、prefix/history側のrate basinに
0向きmean-reversion、coarse position shrinkage、sticky 3-state rate dynamicsが重なり、
physical motionへの追従が鈍る像と最も整合する。

一方、`candidate_gr_with_opposed_prior`群ではtrue `|rate|`が`0.03258 → 0.02933`へ減るのに、
decoded `|rate|`は`0.04330 → 0.06129`へ増えた。これは急なphysical motionへの
under-responseではなく、wrong-GR emission aliasがdecoderを別方向へ駆動する独立経路である。
よって支配的なrate/history-lag経路と、直接的なwrong-GR経路を分ける必要がある。

直前rate errorの符号は最終episode offsetとepisodes `98.4326%`、SSE加重`99.4073%`で
一致した。ただしrate errorはoffsetの微分であり、episode onsetもerror閾値から定義して
いるため、この符号一致は独立な因果証拠ではない。ここで確定できるのは形成時の運動学と
時間順序までである。この段階のdecoded rateはsmoothed posterior meanの差分であって、
hidden rate posteriorそのものではなかったため、parameter帰属は未確定だった。
後続exp408のactual messageではfiltered rateのzero向きunder-responseがSSE`70.36%`、
exclusive forward / backwardがSSE`59.40% / 23.04%`と確定した。

#### Decoder別rate lag

同じrate定義を凍結済みposterior mean、row-wise marginal MAP、global Viterbiへ適用した。
global Viterbiはexp270の`topk_path_1`で、全区間を通した合法なmax-product pathである。
marginal MAPは各行posteriorのargmaxを独立に並べたもので、既にpersistent episodesの
`53.92%`でhard transition違反が確認されている。

| decoder | `|rate error|` far | `|rate error|` near | pooled near/far | near増加episodes / SSE | transition crescendoとのrho |
| --- | ---: | ---: | ---: | ---: | ---: |
| posterior mean | `0.01581` | `0.03365` | `2.2848x` | `79.00% / 85.58%` | `0.6812` |
| marginal MAP | `0.04530` | `0.06853` | `1.5479x` | `61.87% / 59.52%` | `0.5335` |
| global Viterbi | `0.02151` | `0.03196` | `1.6477x` | `65.27% / 56.23%` | `0.8364` |

global Viterbiでもnear rate errorはfarの`1.65x`であり、transition pressureとの対応は
posterior meanより強い。最大の`prior_persists_against_truth_gr`群でも
`0.02208 → 0.03292`、`1.6848x`、transition crescendoとのrho `0.8382`だった。
したがって「真のpathはtransition/history上は正しく、sum-product meanだけが二mode間を
平均してrate lagを作る」という単独原因は反証される。合法なbest path自体が発症前に
truth motionへ追従できていない。

一方、直前16行ではViterbiのabsolute rate errorがposterior meanより平均
`0.00169`小さく、改善はepisodes `49.8433%`、SSE加重`66.3421%`だった。
最大のprior+truth-GR群でも改善SSEは`64.8154%`、prior+wrong-GR群では`79.3064%`である。
しかしepisode全体ではViterbiがmeanを改善するのは`32.2884%`、SSE加重`37.7751%`だけで、
平均absolute rate errorは逆に`0.00188`大きい。よってmax-product化は形成直前の重い
episodeを部分的に軽減するが、長区間の安定な解法にはならない。transition/history
basinが共通のrootで、sum-product path multiplicity / posterior meanは経路依存の増幅器、
という分担が最も整合する。

marginal MAPはfarから既にabsolute rate error `0.04530`とposterior meanの約2.9倍で、
nearは`0.06853`まで増える。これはrow-wise modeのgrid jumpと既知のhard違反によるもので、
物理rateやstable mode IDのcarrierとしては使えない。

#### Viterbi latent rate-stateのstickiness

exp270はrow-level rate path本体の保存を明示的に禁止していたが、各wellのglobal Viterbi
rate-state switch率とjoint top-5各pathのrate-path SHAは保存していた。suffix row数から
switch countを整数へ復元し、最大rounding error`9.27e-13`であることを確認した。

| scope | wells / episodes | top-1 rate zero-switch | switch count median / p90 | top-5 rate hashが1種類 |
| --- | ---: | ---: | ---: | ---: |
| 全wells | 773 wells | **`75.0323%`** | **`0 / 2`** | `75.0323%` |
| persistentなし | 323 wells | `79.8762%` | `0 / 2` | `78.6378%` |
| persistentあり | 450 wells | **`71.5556%`** | **`0 / 3`** | `72.4444%` |
| persistent episodes | 638 episodes | `67.3981%` / SSE`73.2917%` | well値を継承 | `68.4953%` / SSE`77.2609%` |

最大の`prior_persists_against_truth_gr`群でもzero-switch well上のepisodesは
`66.6667%` / 群SSE`72.6546%`、top-5 rate hashが1種類なのは
`69.4444%` / 群SSE`79.6846%`だった。つまり重いoffsetの大半では、少なくとも
max-product解のlatent rate stateは別IDへ頻繁にswitchしていない。むしろ同じrate stateを
well全体で保持しながら、position pathがtranslation gauge上でoffsetを形成・維持している。

persistent wellsはnonpersistentより切替がやや多く、平均countは`1.309 vs 0.560`だが、
episode単位のswitch率とRMSE、posterior-mean rate-error増分、Viterbi rate-error増分、
transition crescendo、pre128 slopeのrhoは
`-0.0298 / -0.0410 / -0.0318 / -0.0068 / 0.0236`である。switchの有無・頻度は
offset severityもonset pressureも説明しない。

したがって「mode IDを保持する」をlatent rate stateへ解釈しても、global Viterbiでは
すでに大半のwellで実現している。それでもoffsetが残るため、必要なのは単一rate IDの固定
ではなく、absolute position basinへ属するsum-product mass、そのbasin内rate moment、
predictive / filtered / smoothed間の更新を追跡することである。exp270にはrate-path本体と
rate posteriorが保存されていなかったため、この点が後続exp408の直接監査を必要とした。

#### Rate方向の排他的分解

平均`|rate|`の差が同じepisode内のunder-responseを本当に表すか確認するため、true rateが
非zeroの各行を次の4 classへ排他的に分けた。

- zero-directed under-response: decoded rateは真値と同方向または0で、絶対値だけ小さい
- opposite direction: decoded rateが真値と逆方向
- same-direction overshoot: 同方向だが絶対値が真値より大きい
- tie / boundary: tolerance内の同値またはclass境界

episode内807,710行のうちtrue rateが非zeroの778,966行を分類し、28,744 zero-rate行は
方向比率から除外した。全classのrow countは各区間のmoving rowsと厳密に一致した。

| 群・decoder・ring | zero-directed rows / error量 | opposite rows / error量 | overshoot rows / error量 |
| --- | ---: | ---: | ---: |
| 全体 mean far | `52.50% / 52.19%` | `4.63% / 13.91%` | `39.37% / 33.91%` |
| 全体 mean near | `57.76% / 52.99%` | `16.26% / 27.17%` | `25.37% / 19.85%` |
| prior+truth-GR mean near | **`73.08% / 64.66%`** | `16.73% / 27.21%` | `9.77% / 8.13%` |
| prior+truth-GR Viterbi near | **`64.20% / 59.31%`** | `12.88% / 25.43%` | `14.66% / 15.26%` |
| opposed-prior+candidate-GR mean near | `4.76% / 1.15%` | `1.34% / 2.33%` | **`93.45% / 96.52%`** |
| opposed-prior+candidate-GR Viterbi near | `15.48% / 13.31%` | `1.49% / 4.66%` | **`77.23% / 82.03%`** |

これはendpointだけの差でもない。最大prior+truth-GR群のmean zero-directed row率は
farからnearへ`59.74 → 61.70 → 66.06 → 71.10 → 73.08%`と段階的に増え、
opposed-prior+candidate-GR群のovershoot率は
`54.61 → 77.74 → 86.28 → 87.56 → 93.45%`へ増えた。Viterbiもそれぞれ
zero-directed `40.34 → 46.53 → 54.79 → 60.36 → 64.20%`、
overshoot `41.70 → 57.07 → 68.43 → 71.96 → 77.23%`で、二つの形成様式が
発症へ向けて連続的に強まる。

最大の`prior_persists_against_truth_gr`群では、posterior meanだけでなく合法Viterbiでも
zero-directed under-responseが直前moving rowsとabsolute rate-error massの過半を占める。
これは群平均の見かけではなく、行単位でもposition/rate grammarの0向き追従不足が支配する
直接証拠である。ただしmeanではopposite directionも直前`16.73%`、誤差量`27.21%`まで増え、
純粋なshrinkageだけでは全誤差を説明しない。

`candidate_gr_with_opposed_prior`群は反対で、直前mean rowsの`93.45%`、rate-error massの
`96.52%`がsame-direction overshootである。Viterbiでも`77.23% / 82.03%`を占めるため、
この群はsum-product平均化ではなく、GRに整合するwrong/high-rate joint pathへの過走である。
一方、priorとwrong GRが同方向の群ではnear zero-directedが`68.99%`であり、candidate-GR群を
一括して「GRによるovershoot」とも呼べない。prefix basinとGR方向の組合せが発火形を決める。

以上から、全体を一つのstable mode IDで説明するより、

1. prior-aligned側のzero-directed rate under-response
2. opposed-prior + candidate-GR側のsame-direction overshoot
3. neither / ambiguous群のovershoot・opposite混合

へ分ける方が証拠に合う。この保存artifactにはhidden rate state massがなかったが、
後続exp408でpredictive / filtered / smoothed rate massを保存し、forward hysteresisを
主因、backward reversalとmultiplicityを増幅器として分離した。

#### GR missingnessのonset timing

rate lagの発症がraw GR missing gapへの突入で起きる可能性を、同じnon-overlapping ringの
raw-observed row countから監査した。

| ring | missing率 episode mean | pooled missing率 |
| --- | ---: | ---: |
| 256--512行前 | `31.6220%` | `31.5704%` |
| 128--256行前 | `31.1473%` | `31.1363%` |
| 64--128行前 | `31.1214%` | `31.1869%` |
| 16--64行前 | `30.7798%` | `30.7798%` |
| 0--16行前 | `30.3977%` | `30.3977%` |

far / near双方を持つ619 episodesではnear-minus-far missing率の平均`-1.1696 points`、
median`-0.7813 points`で、増加は`45.2342%`、減少は`53.7964%`だった。missing変化と
truth transition crescendo、posterior-mean rate-error増分、Viterbi rate-error増分、
pre128 error slope、episode RMSEのrhoはそれぞれ
`-0.0217 / -0.0275 / -0.0538 / -0.0273 / 0.0461`で、いずれも実質無相関である。

最大の`prior_persists_against_truth_gr`群もmissing率は
`33.6709 → 33.8145 → 32.9363 → 33.2851 → 32.3785%`で増加せず、
missing変化とmean rate-error増分のrhoは`0.0447`だった。
`candidate_gr_with_opposed_prior`群はfar`15.8887%`、near`15.1989%`と低く安定し、
全missing episodeは0である。したがって直接wrong-GR経路はimputed GRだけでなく、
観測されたGRが十分ある状態で発火している。

直前16行が全missingなのは1 / 638 episodes、SSE`2.0581%`だけだった。一方、
missing率がfarから10 points以上増えるのはepisodes `18.9015%`、SSE`24.6968%`、
25 points以上は`5.3312%` / SSE`7.0792%`である。よってmissing増加は一部episodeの
absolute-anchor低下・severity modifierにはなり得るが、rate-lag形成の一般トリガーからは
除外する。これはmissing/imputation qualityの効果そのものを反証するのではなく、
「発症直前にmissingへ入ったからoffsetが始まる」という時間仮説の反証である。

## Position transition の量子化bias

exp209のposition transitionは設定上`sig_p=0.02 ft`だが、実装では
`max(sig_p, 0.35 × step)`なので実効値は`0.1225 ft`である。0.35 ft grid上の5点離散
Gaussianをこの狭さで正規化すると、連続なtransition meanをそのまま期待変位として再現
しない。たとえば真の1行TVT変位が`+0.10 ft`なら、

```text
kernel center mu = +0.10 ft
discrete expected displacement = +0.051293 ft
bias = -0.048707 ft / row
```

となり、0方向へ約49%縮む。負変位では符号が反転する。このbiasを全suffixで、各行の
真のTVT変位をkernel centerに置くtruth-late counterfactualとして監査した。

| 監査 | 値 |
| --- | ---: |
| 対象 | 3,783,989 rows / 773 wells |
| 1行biasの絶対平均 | 0.011553 ft |
| 1行biasの最大絶対値 | 0.048707 ft |
| abs bias > 0.02 ftの行 | 22.5954% |
| well内の累積biasと実prediction errorの相関 median | 0.5915 |
| 同相関が正のwell | 635 / 773 |
| episode開始前128行biasとepisode平均offsetのSpearman | **0.6346** |
| initial-rate mismatchを制御したpartial Spearman | **0.5510** |
| 同bias絶対値とepisode RMSEのSpearman | 0.0790 |
| 同biasとoffset符号が一致 | **85.1097%** |
| SSE加重の符号一致率 | **83.5618%** |

このtruth-centered局所量はtailの大きさより、実offsetが浅側・深側のどちらへrampしたかと
強く対応する。最後にerror 5 ft以内だった時点からepisode開始までを積算しても符号一致率は
85.2665%、SSE加重81.7266%だった。約200--250行の緩やかなonsetとも時間尺度が整合する。
実episodeは正offset 276件 / SSE 36.8457%、負offset 362件 / SSE 63.1543%の両方があり、
kernel-bias符号一致は各`84.78% / 85.36%`だった。一定符号のdatum/indexずれではなく、
rate方向に対してほぼ対称なtransition biasである。

既知のinitial-rate mismatchとkernel biasのSpearmanは`-0.4595`で一部重なるが、同mismatchを
制御してもkernel biasとsigned offsetのpartial Spearmanは`0.5510`残った。rank回帰の説明率は
kernel bias単独`0.4028`、initial-rate mismatch単独`0.1677`、両方`0.4204`であり、
truth-centeredな記述量としては量子化kernelの方がinitial-rate mismatchより強く対応する。
ただし後述の実prefix transition-only反証により、この相関をactual posterior rateを固定した
直接因果効果とは解釈しない。

`mom=0.998`のrate transition自体が持つ0向きmean-reversionも分離した。現在rateが
truth-centeredな内点にあるとした局所期待position biasは
`-(1-mom) × true_rate × dMD²`である。1行絶対平均は`0.0000689 ft`で、position
kernel bias `0.011553 ft`の`0.5966%`にすぎず、これ単独で10 ft級offsetを線形蓄積する
大きさではない。一方、episode開始前128行の累積biasはsigned offsetとSpearman
`0.5744`、符号一致`72.41%`で、position-kernel biasを制御してもpartial Spearman
`0.3857`が残った。signed-offset rank R²はposition kernel単独`0.4028`から、
rate mean-reversionを加えると`0.4916`へ上がり、initial-rate mismatchも加えた値は
`0.4929`だった。

したがってこの局所1行式で見えるmean-reversionは、position量子化と別の小さい
conditional directional tiltを持つ。ただし後述の制御合成HMMでは、数百行の積分と
position kernelとの相互作用により無視できない振幅へ育つため、「局所量が小さいので
大offsetへ寄与しない」とは結論できない。この式はrate-grid境界、emission、decoded
rate分布を含まないtruth-centered局所診断であり、actual-HMMのmomentum介入に代わらない。

GR evidence別でも同じ関係が残る。

| observed GR evidence | pre128符号一致 | SSE加重符号一致 | bias vs mean offset Spearman |
| --- | ---: | ---: | ---: |
| candidate strong | 81.67% | 71.76% | 0.5226 |
| truth strong | 85.25% | **88.77%** | 0.6622 |
| near tie | 88.33% | 93.91% | 0.7537 |

特にraw GRが真値を支持するtruth-strong群でもSSE加重88.77%でtransition biasとoffsetの
向きが一致する。これは「GRは正しいのにHMMが戻らない」52.76%の大群でtransitionとの
相互作用を疑う根拠になるが、実prefix transition-only priorでは同じ符号対応が逆転するため、
position kernel単独のforward原因を確定する根拠にはならない。

ただしrate grid自体に真の動きを表現する状態がないわけではない。41 rate statesから各行で
最適状態をoracle選択すると、期待変位の絶対representation errorは全行平均
`0.001367 ft / row`まで下がる。一方、その補償に必要なtrue rateからのrate shiftは
rate support内でも絶対値median `0.025`、p90 `0.045`である。これは1ステップrate noise
`sig_r=0.002`の12.5--22.5倍、典型的には5--9 rate cellsに相当する。`mom=0.998`かつ
rate transitionが隣接3状態だけなので、HMMは量子化biasを打ち消すrateへ即座に移れず、
GR emissionと競合しながら長いoffsetを形成する。

この遅さは実装式からもepisodeの時間尺度と一致する。rate transitionの1行分散は
`sig_r² × dMD`で、補償shiftへ拡散する局所時間proxyを
`shift² / (sig_r² × dMD)`と置くと、全rate-supported rowsでmedian `156.25 rows`、
p90 `506.25 rows`、p95 `625.00 rows`だった。これは厳密なfirst-passage timeではなく、
mean reversion、時変target、emission、smoothingを省いた記述値だが、実測の
「最後の5 ft以内からepisode開始までmedian 232 rows」「16 rows以内は0件」と同じ桁である。

さらに`mom=0.998`はdMD 1 ftでrate期待値を1行ごとに約0.998倍するため、rate履歴の半減期は
約346行、e-fold時間は約500行になる。量子化biasの補償拡散156--506行、mean-reversion
346--500行、offset onset 232行が独立に同じ数百行スケールへ揃う。これはsticky rate
dynamicsが小さいdirectional tiltやprefix-rate mismatchを長いrampへ変換できることを
支持するが、tiltの主な向きをposition kernelだけへ帰属するものではない。

5点kernelの位相だけを走査する決定論的数値監査では、maximum abs biasは現行
`sigma=0.1225`の`0.048765 ft`から、`sigma=0.20`で`0.002286 ft`、
`sigma=0.245`で`0.000848 ft`へ下がった。5点truncationも含む最小点は、最大bias基準で
`sigma≈0.2325 ft`、位相平均bias基準で`≈0.2365 ft`だった。これはCV選択ではないが、
現行floorが5点離散近似として特に狭く、過去exp327の上限`0.245 ft`が数値的に妥当な
範囲だったことを示す。actual-HMMでsigmaを変える因果介入は別途必要である。

### Position解像度と一次・二次モーメントのtrade-off

`sigma=0.2325 ft`はmean biasを消すが、process varianceも変える。0.35 ft格子の全位相を
一様走査すると、現行kernelの位相平均分散は`0.015006 ft²`、0.2325 kernelは
`0.053920 ft²`で`3.59x`だった。さらに全3,783,989 actual true motions上で測ると、
現行分散は`0.004578 ft²`、0.2325は`0.053720 ft²`で**`11.734x`**へ増える。
actual motion上のabs mean biasは`0.011553 → 0.000103 ft / row`まで下がるが、
これは一次モーメント補正と大幅な拡散強化の複合介入である。

同じ0.35 ft格子で、要求meanを挟む隣接2セルだけにmassを分けるminimum-variance
exact-mean transportを考えると、actual motion上の平均最小分散は`0.005262 ft²`で、
現行の約`1.15x`に留まる。一方、設定`sig_p=0.02 ft`の連続分散`0.0004 ft²`に対しては
なお`13.16x`である。一般にsub-grid meanをgrid上で厳密に表すには非ゼロ分散が不可避で、
現行gridは設定process noiseに対して粗すぎる。

現行floor式はexp205で公開notebook defaultを明示的に保持したもので、偶発的な単位typoを
示す記録はなかった。しかし数値的には、狭いvarianceを守ればmeanが量子化され、meanを
守ればvarianceが増える構造的な離散化問題である。現行式のままgrid stepを
`0.35 → 0.175 → 0.0875 → 0.04375 → 0.035 ft`へ細かくした位相最大biasは
`0.04877 → 0.02438 → 0.01219 → 0.00186 → 0.00023 ft`へ下がるが、position state数は
`1x → 2x → 4x → 8x → 10x`になる。したがってfiner-gridはmean/varianceを分離しやすいが、
full exact HMMのruntime・memoryを大きく増やす。

### 制御合成HMMによる遷移機構の因果分離

truth-late相関だけでなく、exp209と同じ41-state rate transition、0.35 ft position grid、
5点position kernel、forward-filter-backward recursionを使う制御合成HMMを作った。
geologyとtypewell GRは一切使わず、rate-grid上の一定true rateから正しいstateで開始し、
真のpositionを中心とするGaussian emissionだけを与えた。position sigma補正、rate境界行の
再正規化、`momentum=1`を2×2×2の全8通りで512行decodeし、さらにminimum-variance
exact-mean transportの4条件を加えた。全12 variants × 8 scenarios = 96 casesである。
これはOOF parameter選択では
なく、遷移演算だけのmechanism diagnosticである。dMDは1 ftに固定したが、実OOF suffixも
全3,783,989 transitionsでraw dMDのmin / maxがともに厳密に1 ftだったため、この点は
単純化ではなく実データと一致する。

position格子端massは全96ケースで最大`2.2825e-10`、`rate=+0.05/-0.05`の符号反転誤差は
最大`1.0658e-14 ft`で一致した。rate境界を再正規化したneutral-emission条件では
filteredとsmoothedのRMSE差も最大`1.57e-8 ft`に留まり、forward/backward実装の
確率保存不変条件を満たした。最適化したtransition演算をexp270と同じsource/destinationの
素朴な全列挙へ照合した1-step最大差はforward/backwardとも`5.4210e-20`だった。

正しく中心化した2 ft Gaussian position emissionでのfiltered RMSEは次の通りだった。
`P`はposition sigmaを位相minimaxの`0.2325 ft`へ変更、`B`はrate境界source rowを再正規化、
`M`は`momentum=1`を表す。

| true rate | current | P | B | M | P+M | P+B+M |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| +0.025 | 0.296972 | 0.039833 | 0.295346 | 0.235176 | 0.001140 | **0.001018** |
| +0.050 | 0.450709 | 0.080201 | 0.418143 | 0.381468 | 0.007005 | **0.001720** |
| +0.100 | 0.916870 | 0.237125 | 0.860956 | 0.879137 | 0.176544 | **0.116887** |
| -0.050 | 0.450709 | 0.080201 | 0.418143 | 0.381468 | 0.007005 | **0.001720** |

`rate=0.05`ではposition sigma単独で`82.21%`、boundary単独で`7.23%`、
momentum単独で`15.36%`、全3補正で`99.62%`改善した。position sigma補正後に
momentumを1へ変えると`0.080201 → 0.007005 ft`まで下がったため、mean-reversionは
1行の局所biasだけを見るより、数百行の積分とposition kernelとの相互作用で大きくなる。
一方、interior rateでboundary再正規化だけの効果は小さく、実OOFで見えた
quantization-to-boundary couplingは全体root causeではなくtail severity modifierという
位置づけを維持する。true rate自体が境界の`0.10`では全補正後にも`0.116887 ft`残り、
rate support端では別の制約が残る。

`E`をminimum-variance exact-mean transportとして、position readoutの広がりも比較した。
値はcentered 2 ft emissionでの512行平均である。

| true rate | kernel | filtered RMSE | smoothed RMSE | filtered position std | smoothed position std |
| ---: | --- | ---: | ---: | ---: | ---: |
| +0.025 | current | 0.296972 | 0.040654 | 0.423481 | 0.311242 |
| +0.025 | P | **0.039833** | **0.004382** | 0.671457 | 0.481101 |
| +0.025 | E | 0.094589 | 0.015749 | **0.471180** | **0.312654** |
| +0.050 | current | 0.450709 | 0.057524 | 0.471374 | 0.357170 |
| +0.050 | P | **0.080201** | **0.008841** | 0.671527 | 0.481400 |
| +0.050 | E | 0.129908 | 0.018111 | **0.518113** | **0.357674** |

Eもcurrent比でfiltered RMSEを`68.1--71.2%`削減し、一次モーメント不一致が因果要因である
ことをsigma拡大とは別に確認した。PのRMSEはさらに低い一方、filtered position stdも
current比約`42--59%`広い。Eのstd増加は約`10--11%`に留まる。したがってPの追加改善には、
mean bias除去だけでなく、広いprocess massがcentered emissionへ追従しやすくする効果が
含まれる。actual GRではこの拡散がwrong-depth basinへの移動も増やし得るため、
Pのsynthetic優位をそのままOOF採用根拠にはできない。

emissionをneutralにすると、currentの512行目誤差は`rate=0.025`で`-10.3776 ft`、
`rate=0.05`で`-20.5025 ft`まで自力で蓄積した。`rate=0.05`ではPが`-10.0565 ft`、
Eも`-9.9631 ft`とほぼ一致し、P+Mが`-5.0035 ft`、P+B+Mが`-1.1037 ft`へ順に縮めた。
PとEがneutralで同程度まで回復することは、両者が同じposition mean biasを除く証拠である。
したがってGR aliasがなくても
transitionはoffsetの向きとdriftを作れる。ただし正しいemissionがあれば振幅は強く抑えられる。

message別には、`rate=0.05`で以下となった。

| centered emission | predictive RMSE | filtered RMSE | smoothed RMSE |
| --- | ---: | ---: | ---: |
| sigma 2 ft | 0.473991 | 0.450709 | **0.057524** |
| sigma 10 ft | 2.077254 | 2.051431 | **0.512039** |

cleanな単一truth basinでは、正しいrow emissionはforward priorのbiasを少し修正し、
futureの正しいemissionを使うbackward smoothingはさらに大きく修正した。よってactual
truth-strong episodeのoffsetを「backward smoothingは一般に悪い」と説明することはできない。
実データでsmoothing後も別offsetに残るには、transition biasに加えて、別深度でも整合する
構造化GR列、missing/imputation、またはmacro path multiplicityが必要である。

### 実prefix・実geometry上のtransition-only prior

制御合成HMMは正しいrate stateから開始している。これがactual OOFのrate履歴を代表するか
確認するため、各wellの実prefix initial rate、実suffix dMD / dZだけから、GR emissionも
suffix truthも使わずtransition momentを伝播した。truthと保存exp209 predictionはwellごとの
trajectory生成後にだけ結合した。これはprediction scoreではなく、実geometry上でforward
priorの各成分がどちらへ動くかを見る診断である。

| transition-only variant | prior RMSE (診断値) | actual HMM errorとのwell内相関 median | anchorから90 ft以内のrow比 |
| --- | ---: | ---: | ---: |
| current | **39.7839** | **0.5427** | 97.4880% |
| exact-mean position | 95.2364 | 0.4466 | 65.1666% |
| boundary normalized | 39.7895 | 0.5420 | 97.4596% |
| momentum=1 | 42.1575 | 0.5217 | 95.9575% |
| exact-mean + momentum=1 | 83.5932 | 0.4457 | 72.6031% |
| all three corrections | 63.3284 | 0.4396 | 84.6692% |

current prior errorとactual HMM errorのwell内相関median `0.5427`は、prefix rateを持ち越す
forward priorがactual errorへ実際に関係することを支持する。一方、exact-mean化すると
GRなしprior RMSEは`39.7839 → 95.2364 ft`へ悪化した。現行position shrinkageを除いた
episode効果は実offsetとSpearman `-0.4656`、符号一致`32.45%`、SSE加重`18.85%`であり、
truth-centered局所監査とは逆向きだった。長い伝播でfinite position bandを越える影響を
除くため、比較する両trajectoryがepisodeの90%以上とonsetでanchorから90 ft以内にある
239 episodesへ限定しても、Spearman `-0.1343`、符号一致`43.10%`、
SSE加重`22.68%`で、position補正がactual offset方向を説明する証拠にはならなかった。

rate境界再正規化のepisode効果絶対値medianは`0.1133 ft`で、actual offsetとのSpearman
`-0.0304`、符号一致`50.31%`。mean-reversion除去は絶対値median`3.0056 ft`だが、
Spearman `-0.2349`、符号一致`43.57%`だった。全補正のanchor-safe 439 episodesでも
Spearman `0.0401`、符号一致`56.26%`、SSE加重`50.97%`で、actual offset方向との対応は
ほぼ消えた。

この反証は局所式や制御合成HMMの数値結果を否定しない。両者は「正しいrate basinにいる
場合、coarse-grid shrinkageとmean-reversionがdriftを生成する」ことを示す。一方、
実prefixではrate prior自体がfuture geometryに対してstale / misspecifiedであり、現行の
shrinkageがその外挿を抑えるregularizerとして働く。したがってactual offsetの符号を決める
のはtransition定数単独ではなく、filtered rate posterior、GRによるrate-state更新、
backward message、position-rate joint massの相互作用である。

position moment伝播は平行移動不変として有限`±100 ft`position bandを省いたため、
variant全体の絶対RMSEをactual HMM scoreとして解釈してはならない。ただし上記の
anchor-safe subsetでも符号対応が回復しないため、主要な反証は有限band逸脱だけでは
説明できない。最適化版は旧4-loop版と全共通列でby-well最大`2.27e-13`、
episode最大`1.82e-12`まで一致し、full 773 wellsを`317.9 sec`で完了した。

### Prefix prior・GR evidence・position shrinkageのjoint分解

transition-only episode指標、observed-GR evidence、truth-centered kernel監査を
638 episodeの完全一致keyで結合した。current prefix prior errorとactual HMM mean errorは
Spearman `0.6128`、符号一致`75.3918%`で、符号一致481 episodesがepisode SSEの
`85.3208%`を占めた。current priorの絶対errorはmedian `36.7507 ft` / p90
`77.2020 ft`、actual HMMは`13.9152 / 27.9621 ft`である。actual HMMの方がcurrent priorより
真値に近いepisodeは`80.8777%`、SSE加重では`69.9618%`だった。したがってHMMは一般に
prefix外挿の大誤差を新しく作るのではなく大幅に修正しているが、同じ符号の残差を
取り切れないepisodeがtailを支配する。

この関係はobserved GR classをまたいで残った。

| observed GR evidence | episodes | prior vs actual rho | prior符号一致 | SSE加重符号一致 | actual/prior abs比 median |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate strong | 180 | 0.6209 | 75.56% | 86.45% | 0.4950 |
| near tie | 180 | 0.5314 | 71.67% | 89.30% | 0.4214 |
| truth strong | 278 | **0.6392** | **77.70%** | 84.02% | 0.4735 |

wrong-depth GRはcandidate-strong群を固定するが、initial direction自体はGR classだけで
決まらない。特にtruth-strongでもprior alignmentが最も高く、正しいGRがprefix-rate
historyを完全に上書きできないことが主要な未解決現象である。onset時間、
episode開始位置、episode長、raw-GR missing率、prefix sigmaを各四分位に分けても、
prior vs actual Spearmanは全bucketで`0.4888--0.7636`、符号一致は
`65.63--85.63%`に残り、特定の長さ・missing bucketだけの結果ではなかった。

prior方向とGR evidenceを組み合わせると、638 episodesを排他的に次の6群へ分けられる。
Viterbi回復SSE率は、その群のSSEのうち`partial_recovery`または`large_recovery`に入る比率
である。

| 排他的cause bucket | episodes | 全episode SSE内 | Viterbi回復SSE率 | 解釈 |
| --- | ---: | ---: | ---: | --- |
| prior aligned + truth GR | 216 | **48.1910%** | **86.29%** | 最大群。正しいraw GRでもhistory massが残り、多くはより良いglobal pathを保持 |
| prior aligned + candidate GR | 136 | 28.8866% | 82.89% | prefix priorとwrong-depth GRが同じwrong basinを支持 |
| prior aligned + GR near-tie | 129 | 8.2432% | 83.31% | GR識別力不足下でprior/historyが残る |
| prior opposed + truth GR | 62 | **9.1626%** | 75.91% | prefix priorもobserved GRも実offsetを説明せず、beta・imputed GR・path multiplicityの重点群 |
| prior opposed + candidate GR | 44 | 4.5292% | 68.77% | candidate側GRがprefix priorと逆方向のwrong basinを支持 |
| prior opposed + GR near-tie | 51 | 0.9874% | 37.16% | 小寄与の未識別群 |

`prior aligned + truth GR`が最大で、かつそのSSEの86.29%はViterbiで少なくとも部分回復する。
同群はobserved行だけでなく、補間GR行を含む全row evidenceでもSSEの`99.87%`が
truth-strong、missing/imputed行だけでも`92.16%`、prefix static affine calibration後の
observed evidenceでも`90.04%`がtruth-strongだった。したがって最大群を
「observed GRは正しいが補間GRまたはstatic calibrationがwrong basinを支持した」とは
説明できない。
したがって最大の未解決原因は「wrong pathがstate spaceから消えた」ことではなく、正しい
row emissionとより良いglobal pathが存在しても、sum-productのhistory/path massが
posterior meanをprior側へ残すことである。一方、`prior opposed + truth GR`の9.16%は、
transition-only directionとobserved raw GRのどちらでも説明できず、backward message、
position-rate joint multiplicityを直接保存しない限り原因を確定できない。この群も
全row evidenceはSSEの`100%`がtruth-strong、imputed-onlyも`88.48%`がtruth-strongで、
affine observedも`91.23%`がtruth-strongだった。imputation aliasとstatic affine mismatchは
主説明から除外できる。

ただし「Viterbi回復可能」を「正しい別modeへ完全に移る」と読み替えてはいけない。保存
row predictionからepisode平均の符号を再集計すると、marginal MAPはposterior meanと
`99.5298%`同符号で、平均error 5 ft以内は`2.1944%`のepisodes / SSE`0.4138%`だけだった。
global Viterbiは平均errorをposterior meanより縮めるepisodeが`60.6583%`、SSE加重
`81.9481%`ある一方、符号はなお`77.4295%`のepisodes / SSE`74.8571%`で同じで、
5 ft以内まで解消するのは`21.4734%` / SSE`20.0658%`に留まった。最大の
`prior aligned + truth GR`群でもViterbiは75.46%で同符号、5 ft以内は22.22%である。
したがってstable mode IDやhard Viterbiへの置換だけで全offsetが直る像は否定され、
continuousなrate/history biasと同一符号basin内のmass配置も診断対象に残る。

position shrinkageの役割はprior directionで明確に分岐した。ここで効果は
`current prior - exact-mean prior`であり、正ならcurrent shrinkageが正方向へ動かした量を
表す。

| regime | episodes | episode SSE内 | position効果 vs actual rho | 効果符号一致 | SSE加重符号一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| current priorとactualが同符号 | 481 | **85.3208%** | **-0.7476** | 10.81% | 4.92% |
| current priorとactualが逆符号 | 157 | 14.6792% | **+0.7243** | 98.73% | 99.83% |

position効果とcurrent prior error自体はSpearman `-0.9226`で、shrinkageはほぼ常に
prefix-rate外挿をanchor側へ戻す。したがって支配的な85.32% SSE群ではshrinkageは
actual offsetを作るのではなく抑え、少数の14.68% SSE群ではactual offset方向へ寄与する。
exact-mean priorの絶対error medianは`91.4905 ft`でcurrentの`36.7507 ft`より大きい。
position kernelは全体root causeでも無関係でもなく、rate basinによって符号が反転する
regime-dependent modifierである。

rate側も単純な単独効果ではなかった。current position shrinkageを残したまま
`momentum=1 + rate境界再正規化`へ変えるtransition-only priorは、episode平均error絶対値
medianを`36.7507 → 29.1207 ft`へ下げ、`65.9875%`のepisodes / SSE`72.4382%`でcurrent
priorより真値へ近づいた。ただしcurrentとの差のactual offsetに対するSpearmanは
`0.1506`、符号一致`58.31%`と弱い。boundary単独は効果median`0.1133 ft`、
momentum単独は`3.0056 ft`で方向対応がなく、両者の組合せでrate massが境界へ達したときの
interactionと解釈する必要がある。position exact-meanまで含む全補正は絶対median
`45.4590 ft`へ再悪化した。したがってrate dynamicsのjoint correctionはprior magnitudeを
減らす候補だが、offset方向のroot causeを単独で説明せず、actual posterior介入の前に
edge massとfiltered rateを観測する必要がある。

truth-centered pre128 kernel biasはactual offsetとSpearman `0.6346`、current prior、
rate mismatch、observed GR NLLを制御した記述的partialも`0.4850`残る。しかしこの量は
future true rateをsource rateに置くため、actual HMMが保持したrate massを表さない。
実prefix interventionの符号が逆になる以上、この高相関は「正しいrate basinなら起きる
transition drift」と「truth motionに対するprefix lag」を混ぜたtarget-late proxyであり、
implemented HMMにおけるposition kernelの平均因果効果としては使えない。

以上から現時点で最も整合する因果順序は、

1. prefixから持ち越したrate distributionが大きいtransition-only driftの向きを作る
2. GR emissionとfilteringが通常はその大半を修正する
3. candidate-strongではwrong-depth GRが残差を固定する
4. truth-strongでは正しいGRでもrate/history massを完全に再捕捉できない
5. position shrinkageは支配群では保護、小さいprior-opposed群では増幅として働く
6. backward messageとpath multiplicityが最終的なposterior-mean residualを決める

である。次のactual passではposition variantを先に走らせず、同一のactual filtered
source-rate massに対するcurrent kernelとexact-mean kernelの1-step期待変位差を保存する。
これならrate posterior feedbackを固定し、position momentだけを直接分離できる。

ここで「prefix priorが方向を持つ」は「last-30 rate estimatorの選択ミス」と同義ではない。
exp268はactual exact HMMのinitial-rate windowだけを`32 / 64 / 128 / 256`へ変えたが、
423 / 773 wellsで全windowのrateが同一、rate spread p90は`0.02`だった。best direct
w128もexp209比`0.0427 ft`改善、5候補whole-well oracleでも`0.0973 ft`のheadroomに
留まった。したがってstatic windowの選び直しではpersistent offsetを解消できず、
原因はprefixからunknown suffixへ固定rateを外挿する構造、suffix内の非定常rate、
GRによるrate posterior更新不足まで含む。

以上から、主問題は単なるgrid support不足でも、position kernel単独でもなく、

1. prefixから持ち越したrate priorとfuture geometryのずれ
2. coarse gridと狭いprocess noiseの不整合によるsub-grid motionの一次モーメントshrinkage
3. mean-reversion、stickyなrate transition、補償rateの境界penalty
4. GR alias / reference mismatchと、rate posteriorを更新するforward filtering
5. 構造化evidenceを集約するbackward messageとsum-product smoothing

の相互作用である。過去の`exp327_time_varying_position_sigma_floor_audit`は量子化問題を
仮説化していたが、親chain不成立により未実装・未実行であり、negative resultではない。
今回の全OOF監査では、同軸は条件付き機構として残る一方、actual rate posteriorを観測せず
positionだけを補正する根拠は弱まった。

この点に見落としがないか、全397 experiment configsと全Python sourceも機械的に再走査した。
HMM transition関係は61 configsで、同じmappingに数値として記録された値は
`sig_p=0.02`が59 / 59、position `step=0.35`が54 / 54、`momentum=0.998`が61 / 61だった。
Python実装内の`sig_p`数値literal 67件も全て`0.02`で、非default literalは0件である。
position変更を設計したのはexp327、momentum変更を設計したのはexp326だけだが、両方とも
親chain不成立でclosed-without-runだった。したがって、既存完了実験にactual position sigma /
grid / momentum介入が埋もれている可能性は除外でき、actual-HMM介入が未実行という判断を
repository全体で確認した。

### exp355 rate-mean介入との交差検証

position sigmaそのものを変えた完了実験はないが、exp355はGR・sigma・grid・`sig_r`・
`sig_p`・momentumを固定し、transitionのrate meanだけをexp226 K16 geometry scheduleへ
変更している。保存済みexp355 OOFを、exp209からtruth-lateに固定した638 persistent episodesへ
再結合した。

| scope | exp209 RMSE | exp355 RMSE | SSE差 |
| --- | ---: | ---: | ---: |
| 全3,783,989 rows | 11.9383 | 11.2920 | -56,812,683 |
| exp209 persistent 807,710 rows | 24.7831 | **20.9282** | **-142,328,004** |
| それ以外2,976,279 rows | **3.8102** | 6.5765 | +85,515,321 |

exp355は元のpersistent episode SSEを`28.6896%`削減したが、元々persistentでなかった行に
大きい新規errorを作った。episode単位では384 / 638が改善し、RMSE gain median
`0.2893 ft`だった。GR evidence別のpooled SSE削減率はcandidate-strong `15.88%`、
near-tie `19.75%`、truth-strong **`37.59%`**で、raw GRが正しいのにHMMが戻らない群ほど
rate-mean介入が効いた。

pre128 kernel-biasとepisode mean errorのSpearmanは`0.6346 → 0.6139`、SSE加重符号一致は
`83.56% → 70.40%`へ下がった。K16 scheduleは量子化・rate履歴問題を部分的に補償するが、
wellごとに必要な補償量と向きを安全に推定できず、別のoffsetを生成する。

これは「transition dynamicsを動かせば既存offsetが大きく回復する」という因果方向を
支持する。一方、exp355はrate meanを変更する複合介入なので、position-kernel shrinkage、
momentum、forward / backward messageの個別寄与までは分離しない。また、元episodeだけの
回復を理由にexp355を採用することもできない。

## 固定grid・rate support・初期条件の反証

### TVT grid

| 監査 | 値 |
| --- | ---: |
| 真値がTVT grid外 | 135 / 3,783,989 rows |
| 該当well | 1 / 773 |
| 全SSEに占めるgrid外SSE | 0.001491% |
| posterior meanがgrid端3.5 ft以内 | 0 rows |
| 真値がtypewell TVT範囲外 | 0 rows |
| posterior meanがtypewell TVT範囲外 | 0 rows |

固定`last_tvt ± 100 ft` bandが主因という仮説は棄却できる。
typewell GRの`np.interp`端値外挿も実評価行では発火しておらず、flat boundary emissionは
原因ではない。

### Rate support

- true U-rateがrate span外となる割合が1%以上のwellは86 / 773。
- raw HMM RMSE 30 ft以上の21 wellsのうち、rate span外1%以上は5 wells。
- rate span外とwell RMSEのSpearmanは0.2129。
- persistent episodeでrate span外行が10%以上なのは18 / 638 episodesで、
  episode SSEの6.3949%。
- episodeのrate span外率とRMSEのSpearmanは`0.3264`、episode rowsを制御した
  partial Spearmanは`0.2514`。
- rate span外0の440 episodesはpooled RMSE `22.5577`、0--1%の117 episodesは
  `26.4995`、1--10%の63 episodesは`26.3760`、10%以上の18 episodesは`37.0486`。
- exp270 global Viterbi top-1は全773 wellsでrate-grid edge occupancyが0。
- global Viterbiのrate pathは580 / 773 wellsで全suffix一定であり、RMSE 30 ft以上の
  21 wellsでも17 wellsが一定だった。ただしrate-switch率とwell RMSEのSpearmanは
  0.0766に留まる。
- initial rateによりrate spanが既定`0.10`より広がるのは52 / 638 episodesで、
  episode SSEの5.6605%。同群のpooled RMSEは`22.0266 ft`でpersistent全体
  `24.7831 ft`より低く、rate spanとepisode RMSEのSpearmanは`-0.0563`だった。
  可変spanによる41-state rate gridの粗粒化もtailの主因ではない。

rate spanやrate境界は全体の主因ではないが、severity modifierとしては無視できない。
特に10%以上がspan外の18 episodesは少数ながら高RMSEである。ただし全episode SSEへの
寄与は6.39%なので、残り93.61%の説明にはならない。
constant-rate joint pathが多いことはtransition persistenceの強さを示すが、それだけでは
good/bad wellを識別しない。

一方、「真のrateがgrid外か」と「position量子化biasを補償するrateがgrid境界か」は
別問題だった。真のrateのnearest stateが境界になるのは全行`0.5247%`で、10%以上該当する
19 episodesのSSE寄与は`6.5689%`に留まる。しかし、5点position kernelの期待変位を真値へ
最も近づけるoracle補償rateが境界になるのは全行`6.2407%`、10%以上該当する
211 episodesで、episode SSEの**`40.4812%`**を占めた。truth-strongだけでも102 episodes、
同class SSEの46.12%、全episode SSEの26.45%である。

exp209のrate transitionは境界sourceからgrid外へ向かう確率を再正規化せず捨てる。
oracle補償stateが境界の行では、捨てられるoutward probabilityはmedian `0.06`、
すなわち1行ごとにsource massが`0.94`倍、log-spaceで約`-0.0619`の暗黙penaltyを受ける。
このcounterfactual edge log-mass-loss合計とepisode RMSEのSpearmanは`0.6176`だった。
単なるepisode長、true-rate span外率、補償shift、拡散時間をrank制御しても、補償edge率と
RMSEのpartial Spearmanは`0.3052`残る。global Viterbiの実rate edge occupancyが0であることも、
必要な補償stateをdecoderが避ける像と整合する。

これはtruth-lateに各行最良の補償rateを置いた局所診断であり、保存されていないposterior
rate-edge massそのものではない。それでも、主問題を「41 statesに真rateがない」から、
「position量子化を打ち消すrateがしばしば境界へ押し出され、sub-stochastic boundary
transitionに罰せられる」へ具体化する。特に旧episode SSEの約40.5%で、この
quantization-to-boundary couplingがseverity候補になる。

### Initial prior

- episode開始位置のsuffix fraction中央値は0.4898。
- suffix先頭10%以内に始まるepisodeは59 / 638で、episode SSEの20.1649%。
- posterior mean RMSEはprediction startから50 ft未満で`0.8971 ft`、50--100 ftで
  `1.5000 ft`、100--250 ftで`2.6826 ft`、1000 ft以降で`13.1366 ft`。

さらに773 wellsの実position gridとrate grid上で、exp209の初期priorをそのまま再構成した。

| 初期prior / 1-step監査 | 値 |
| --- | ---: |
| position prior mean bias 最大絶対値 | `6.94e-17 ft` |
| position prior variance | 全well `0.5625 ft²`（`start_sig²`と一致） |
| rate prior mean bias 最大絶対値 | `4.00e-7` |
| rate prior variance median | `1.00e-4`（`r0_sig²`と一致） |
| 初期rate priorのedge mass median / p99 / max | `4.57e-12 / 8.03e-5 / 8.03e-5` |
| rate内部遷移の意図meanとの差 最大絶対値 | `3.47e-18` |
| probability floor / cap発火 | `0 / 0`（31,693 source rows） |
| 初期prior加重のrate境界mass loss median / max | `2.74e-13 / 4.01e-6` |
| 同境界処理による条件付きrate mean誤差 最大 | `1.81e-7` |

position priorは数値精度内で無偏で、rate priorの離散化誤差と初回境界lossも無視できる。
また全suffixのraw dMDが1 ftなので、各wellの1-step rate kernel監査は全行で使われる局所式と
同じである。内部rate遷移は`mom=0.998`で意図したmeanを機械精度まで正しく再現し、
floor/capも発火しない。したがって、初期anchor、初期rate prior、rate内部遷移の
確率式・離散化はoffsetのroot causeから除外できる。rate境界penaltyは初期時点ではなく、
position量子化を補償するrateへposterior massが後から押し出された場合のseverity modifierである。

episode開始前128行のtrue U-rateとprefix initial rateの差は、episode誤差の符号とは
Spearman `-0.4095`で対応した。future rateがprefixより大きいとHMMが浅側へ遅れる、という
方向性は物理式と整合する。一方、rate差の絶対値とepisode RMSEのSpearmanは`0.0121`、
episode内でも`0.0101`しかない。initial-rate mismatchはramp方向のseedにはなるが、
offsetの大きさやtail severityの主因ではない。ここでいうmismatchはprefixからfutureへの
物理rate変化であり、上記のinitial prior離散化biasとは別物である。

### MD step clamp

exp209はtransition内で`dm = max(raw dMD, 1.0)`とするが、全773 wells /
3,783,989 suffix transitionsでraw dMDのminimum / maximumはともに厳密に`1.0 ft`だった。
1.0以外、1.0未満、非正のstepはいずれも0行で、persistent 807,710 rowsも例外0である。
このclampは現データ上完全なno-opであり、dMD不規則性もoffset severityを説明しない。

### Numerical / replay

- exp209 HMM cacheのdecompressed SHAはexp205 referenceと完全一致。
- exp270の再生成posterior meanは保存exp209とmax / mean差`0 / 0 ft`。
- exp270の3,783,989 rows、well / row key、finite coverageは全件一致。

したがってparallel execution、CSV merge、float serializationの偶発差がoffsetを作った可能性も
除外できる。exp391の0.35 ft parity failureはexp391固有実装の問題であり、exp209 baselineの
不安定性を示すものではない。

exp209はforward / backward log-messageを行ごとに正規化せず`float32`で保持するため、
絶対log値の増大による丸めも別途stressした。同じ41-state rate transitionと5点position
transitionを使う512-step制御HMMで、正規化float64、正規化float32、exp209型の
未正規化float32を比較した。さらに全state共通のemission log定数`-3.5 / row`を加えた。
これはexact posteriorを変えないが、未正規化messageの絶対値を約`1,824`まで増やし、
exp270保存log-likelihoodのworst `-1,691.9`よりやや厳しいscaleにする。

| precision stress | filtered mean最大差 vs float64 | smoothed mean最大差 vs float64 |
| --- | ---: | ---: |
| 未正規化float32、通常scale | `3.24e-6 ft` | `3.36e-6 ft` |
| 未正規化float32、worst-scale stress | `1.52e-4 ft` | `1.48e-4 ft` |

正負rateのsmoothed sign-symmetry誤差も全stressで最大`1.81e-4 ft`だった。実offset
`10--61 ft`より5桁小さく、未正規化float32 messageはroot causeではない。この検査は
実well全件をfloat64再decodeしたbitwise parityではなく、vectorized sparse log recurrenceの
制御stressであるため、個別near-tie path順位への微小影響までは否定しない。しかし、
観測された符号一貫persistent offsetの振幅を生成する数値機構としては強く反証できる。

保存posterior meanの`float32`化についても、実3,783,989値
（範囲`10,047.468--12,888.823 ft`）でULPは最大`0.0009765625 ft`、round-to-nearestの
最大誤差上限は`0.00048828125 ft`だった。10 ft閾値の2万分の1未満で、CSV保存精度も
persistent offsetを作れない。

## Missing GR と sigma

- raw GR missing率50%以上のepisodeは146 / 638。
- それらのepisode SSE寄与は23.6418%。
- missingは重要なstress軸だが、全体の主因ではない。
- exp269でmissing emissionを完全neutral化すると13.348へ悪化した。
- exp358のmissing-distance downweightも12.0126へ悪化、0/5 folds。
- exp346のfinite-only sigmaは13.2950へ悪化した。
- exp398のglobal sigma 1.3倍も12.7107へ悪化した。

補間GRとzero-fill由来sigmaは奇妙な設計ではあるが、単純に外す、観測行を強める、
全体を弱める、のいずれも改善しない。現行sigmaは、真の校正というよりwrong-mode固定を
抑える保護的temperingとして働いている。

prefixの有限GRからexp209内で既に計算されているstatic affine `a,b`をtypewell GRへ
適用し、sigmaとtrajectoryを固定したcounterfactual NLLも監査した。candidate-strongの
episode SSE比は33.42%から36.97%へ増え、truth-strongは57.35%から55.60%だった。
evidence classが変わるepisode SSEは13.28%あるが、affineでwrong-depth evidenceが消える
方向ではない。したがって、exp209が既定で`cal_a_use=1, cal_b_use=0`としていることも
主因ではない。この値はHMM再decodeではなく固定trajectoryの尤度帰属である。

Gaussian emissionの`z² <= 600` capは、638 persistent episodesの観測済みGRにおいて
truth側・candidate側とも発火0行だった。極端値clipがtruth/candidateの識別を潰している
わけでもない。

773 typewell CSVのGR欠損も0行であり、typewell側の`ffill/bfill`はno-opだった。
TVT欠損・重複・元順序の非単調も全て0だった。reference GRの欠損補完や
`np.interp`への不正なTVT軸がflat motifを作る原因ではない。

### GRの自己相関とevidence重複

exp343はknown-prefix GR residualのlag 1--20 ACFから有効相関長`tau`を推定した。
outer-train foldのraw `tau`中央値はfull prefixで`9.7714--10.0400`、last-512 rowsで
`24.2583--25.1728`だった。特に末尾では、隣接行のGR evidenceを独立サンプルとして
積み上げると、実効情報量より大きい尤度差を作る余地がある。これは一度できたparallel
offset basinを、forward / backward双方の長いGR列が強化する機構として整合する。

この一般的なACF監査とは別に、638 persistent episodes内で、真値TVTと誤posterior-mean
TVTの観測GR NLL差を行方向に直接監査した。隣接する観測行でNLL差の符号が同じ割合の
episode中央値は`0.8585`、NLL差のlag-1相関中央値は`0.8827`、lag 1--20のpositive-sequence
記述的IAT中央値は`23.9247`だった。exp343のtail `tau≈24--25`と独立にほぼ一致する。
したがって、GR evidence重複は一般論だけでなく、実際のoffset episode内で発生している。

| observed GR evidence | wrong candidate支持run median / p90 | truth支持run median / p90 | lag-1相関 median | IAT median |
| --- | ---: | ---: | ---: | ---: |
| candidate strong | 29.5 / 66.1 rows | 16.5 / 49.1 | 0.9119 | 24.8654 |
| truth strong | 12.0 / 45.0 rows | 19.0 / 60.9 | 0.8998 | 25.6218 |
| near tie | 9.0 / 24.2 rows | 9.0 / 24.0 | 0.7883 | 18.7279 |

candidate-strong群ではwrong candidateを支持する連続runが長く、GR aliasがsticky basinを
固定する説明と整合する。truth-strong群では逆にtruth支持runが長いのにoffsetが残り、
transition/history massが正しい反復evidenceに抗していることを再確認する。
wrong-candidate支持run長とepisode RMSEのSpearmanは`0.2857`、同run長とobserved NLL合計は
`0.2652`に留まる。したがって自己相関はbasin固定の増幅因子だが、offset severityを単独で
決めるroot causeではない。

ただしexp343で事前固定したwell別`tau_eff` scheduleは、joint-evaluableが
`295 / 773`、fallbackが`478 / 773`だった。clip上限4へfullで`99.7413%`、tailで`100%`
張り付き、full/tail安定性も判定不能、stable foldは`0 / 5`だった。そのためStage 1 HMMは
未実行である。このFAILが示すのは「自己相関がない」ことではなく、current-well固有の
安全なtempering量を当該contractでは識別できないことである。

さらに、exp398の一律1.3倍sigmaはexp209比`+0.7724 ft`悪化し、exp305を含む一律tempering
系も採用根拠を作れなかった。したがって自己相関はwrong-basin evidenceを増幅する
修飾因子として有力だが、単独のroot causeや「全wellのemissionを一律に弱める」という
処方にはできない。次のmessage診断では、row NLLだけでなく、連続区間でGR emissionが
basin log-oddsへ加えた増分と、その後のbackward増分を別々に積算する必要がある。

### Row marginalの二峰性

exp236では別親exp221のexact HMM posteriorを監査し、明示的な二峰rowは
`35,399 / 3,783,989 = 0.9355%`、mean-in-valleyは`0.1792%`、mode switchは17件だけだった。
これは「全行で明瞭な二峰posteriorが発生する」という単純像には否定的である。一方、
exp209のpersistent offsetは約200--250行かけてrampし、top-K joint pathも一グリッドの
局所摂動で埋まる。macro basinの総質量競合は、各行marginalで常に二つの鋭いpeakを作るとは
限らない。親decoderも異なるため、exp236をexp209のbasin mass仮説の反証には使えない。

## 既存介入から分かること

| 軸 | 実験 | 結果と含意 |
| --- | --- | --- |
| absolute geometry unary | exp279 | exp209より1.9023 ft改善するがexp263に届かず、tail悪化。履歴offsetは実在するがanchorも誤る |
| residual-offset HMM | exp281 | episode 551→530、512行回復改善。一方RMSE 9.8274でtail失敗 |
| initial rate window | exp268 | best direct +0.0427 ft、oracle +0.1024 ft。初期rate単独の寄与は小さい |
| position kernel量子化 | exp327 | 問題を設計段階で指摘したが、親chain不成立により未実装・未実行。全397 config / source censusでも完了actual介入0 |
| geometry rate schedule | exp355 | exp209より0.6463 ft、5/5 folds改善。ただしhidden-likeとworst +52.74 ftで失敗 |
| prefix-rate-only相当 | exp362 | 0.7766 ft改善。ただし3/5 folds、worst +52.74 ft。rate dynamicsは平均的に効くが安全でない |
| rate transition noise | exp338 | well別proxyが全well`sig_r=0.004`へclipし、exp209比2.1241 ft悪化。単純な全well拡散高速化は不支持だが、proxyが量子化に支配され因果分離にはならない |
| robust emission | exp374 / exp389 | Student-t +0.2178、Huber +0.0855だがwrong-mode tailを除去できない |
| stronger observed GR | exp307 / exp346 | 1.36 ft前後悪化。GR過信がsmoothingで全系列へ伝播 |
| weaker GR | exp398 | 0.7724 ft悪化。全well一律temperingも不適切 |
| GR自己相関 | exp343 | raw tauはfull約10、tail約24--25。ただし478/773 fallback、clip張り付きでwell別温度を識別できずStage 1未実行 |
| sticky GR reliability | exp363 | bad10 AUC 0.6076、circular差+0.0236の弱い識別signalはあるが、weak mass 58.94%で広すぎStage 1不適格 |
| affine calibration | exp345 | subset平均+0.1695だがworst +9.35、hidden-like未完備 |
| explicit registration offset | exp365 | real NLLは改善するがcircular controlの方が大きく、registration signalと断定不可 |
| reset trigger | exp366 | AUC 0.5000、発火率0.00118%。GR-only reset検出は失敗 |
| future-GR branch選択 | exp283 / exp284 | proposal信号は弱く存在するがsafe baseを高率で誤廃棄し、real GRはshuffledにも勝てない。target-free recovery判定は未確立 |
| row marginal bimodality | exp236 | 別親exp221で二峰row0.94%。明示的二峰の直接decoder置換は不支持だが、exp209 macro-basin massの反証ではない |
| posterior mode paths | exp270 | meanが全体最良だがpersistent episodeではViterbi回復headroomが大きい。ただしexact top-5のepisode datum spanはmedian 0 / p90 0.0010 ft、best-of-5追加改善0.0009 ftで、top-K rankは別macro modeを運ばない |
| stable mode ID | exp391 | 16 wells / 19 eventsのみ、15 unresolved、全candidate fallback。parity・normalization・runtimeもFAILし反証にならない |

## exp391 をどう読み直すか

exp391は「mode id保持を試して否定した」実験ではなく、限定的な実装試験がfail-closedした
記録として扱うべきである。

- 対象は638 persistent episodesではなく19 selected events。
- HMM-supported判定は1 / 19だが、15 / 19がunresolved。
- same-pass parity最大0.35 ft、normalization誤差2.46e-5。
- full換算241.68時間。
- 78,866 candidate rowsは全てsaved exp209へfallback。

したがって「wrong mode仮説が違った」とは言えない。今回の観測済みGR NLL分解により、
wrong GR modeは全SSEの約30.7%で直接支持された。一方、より大きい約52.8%ではGRが真値を
支持しており、exp391が狙った単純なno-switch mode保持だけでは原因を表現できない。

## 証拠強度別の原因判定

| 判定 | 要因 | 根拠 |
| --- | --- | --- |
| 強く確立 | persistent parallel-offsetがSSEを支配 | 21.35%のrowsが全SSEの91.99%、符号一貫性median 1.0。12定義でもSSE 73.74--98.10% |
| 実装式から確立 | position transitionのtranslation gaugeと再anchor不在がoffsetを保持 | grid内部では`p_t,p_{t-1}`の同時+cでtransition不変。edge occupancy 0。形成期abs slopeは固定期の9.43倍、episode内true/candidate rate rho 0.908 |
| 反証 | truth pathのhard state/transition support不足がoffsetを開始 | local illegal率はpersistent / nonpersistent=`0.01424% / 0.01431%`。onset前128行にbreakがあるのはepisodes 2.35% / SSE 3.45%、隣接rate連続性による追加break 0 |
| actual messageで確立 | onset前のsoft transition grammar mismatchがrate lagを発火 | 旧truth-late readoutに加え、exp408でfiltered rate zero向きunder-responseがrows70.91%、SSE70.36%、511 dominant episodes |
| actual messageで確立 | onset直前にdecoded rateがtrue rateへ追従できず、datum差を積分 | absolute rate errorはfar`0.01581`からnear`0.03365`へ`2.28x`。exp408のtransition変位誤差とoffsetはepisode Spearman0.5693、SSE加重符号一致90.22% |
| 反証 | 一般的な真のrate acceleration / hard curvature spikeが形成トリガー | true accelerationはfar`0.00850`からnear`0.00813`へわずかに低下。増加はepisodes 47.82% / SSE 45.88%、transition crescendoとのrho 0.1087 |
| actual messageで確立 | 最大のprior+truth-GR群ではzero-directedなrate under-responseが発火 | 旧trajectoryに加え、exp408ではtruth-strong GR群でもexclusive forward / backward / multiplicity / supportが群内SSE45.22% / 26.58% / 14.95% / 10.85% |
| 行単位で強く確立 | 最大prior+truth-GR群の形成直前はzero-directed under-responseが支配 | posterior meanでmoving rows 73.08% / absolute rate-error mass 64.66%、合法Viterbiでも64.20% / 59.31%。群平均だけの見かけではない |
| 行単位で強く確立 | opposed-prior+candidate-GR群はwrong-rate overshootの別経路 | 形成直前のsame-direction overshootはposterior meanでrows 93.45% / error mass 96.52%、Viterbiでも77.23% / 82.03% |
| 反証 | raw GR missing gapへの突入が一般的なonset trigger | missing率はfar`31.62%`からnear`30.40%`へ微減。変化とmean rate-error増分rho`-0.0275`、transition crescendo`-0.0217`。最大prior+truth-GR群も`33.67→32.38%` |
| subset severity modifier | onset前missing率の大幅増加 | far→nearで10 points以上増えるepisodes 18.90% / SSE24.70%、25 points以上5.33% / SSE7.08%。near全missingは1 episode / SSE2.06% |
| 反証 | rate lagがsum-product posterior meanの平均化だけで生じる | 合法なglobal Viterbiでもabsolute rate errorはfar`0.02151`からnear`0.03196`へ`1.65x`、65.27% episodesで増加し、transition crescendoとのrho 0.8364。最大prior+truth-GR群でも`1.68x` |
| 強く支持、経路依存 | sum-product mass / posterior meanが重い形成期のrate lagを増幅 | nearではViterbiがmeanよりepisodes 49.84% / SSE 66.34%で改善。ただしepisode全体では改善32.29% / SSE37.78%に逆転し、単純Viterbi置換は不可 |
| 条件付き機構を因果確立、actual寄与は未分離 | coarse-grid position kernelの一次モーメント不一致 | truth-centered pre128 biasはsigned offsetとSpearman 0.6346、符号一致85.1%。正しいrate開始の制御HMMでは補正が改善。一方、実prefix transition-onlyではexact-mean化が39.78→95.24 ftへ悪化し、効果rho -0.4656 / 符号一致32.45% |
| 条件付き機構を因果確立、actual寄与は未分離 | rate mean-reversion | truth-centered局所量は符号一致72.4%、kernel制御後partial 0.3857。制御HMMでは除去が改善するが、実prefix transition-onlyの効果はrho -0.2349 / 符号一致43.57% |
| 強く確立 | sticky rate dynamicsが数百行のrampを作れる | 補償時間proxy median156/p90 506行、momentum半減期346行、実測onset median232行 |
| 強く支持 | 補償rateの境界penaltyがtailを増幅 | 補償edge>=10%の211 episodesがSSE40.48%、調整後partial 0.3052。制御HMMではinteriorの単独効果は小さく、true edgeで残差大 |
| 強く確立 | wrong-depth GR matchingが一部を固定 | candidate-strongが全SSEの30.74%、raw GRはcandidate側へ明瞭に近い |
| 強く支持、経路依存 | wrong-GRは一部でinitiator、多くの重いepisodeでは形成後の増幅器 | transition負荷上昇415 episodes中、直前raw GRがtruth側なのはepisodes 54.94% / SSE条件付66.03%。最大prior+truth群は60.96% / 65.00%。一方opposed-prior+candidate-GR群ではcandidate側が70.37% / 73.82% |
| 強く確立 | GR evidenceが系列相関で反復加算される | episode NLL差lag-1相関0.883、IAT23.9行、exp343 tail tau約24--25 |
| 強く確立 | candidateとtruth datumの間のGR emission landscapeが非凸 | truth終点が良いsubsetでもpointwise barrier median 61.82 NLL、20超がSSE95.32%。constant datum補正終点が良いsubsetでもmedian 41.46、20超がSSE83.48% |
| 強く確立 | posterior meanより良いglobal pathは大部分で残るが完全なmode脱出ではない | persistent row-wise decoder oracle15.23 vs mean24.78、Viterbi RMSE recoverableが全SSEの76.19%。ただしViterbi episode平均は77.43%で同符号、5 ft以内はSSE20.07% |
| 強く確立 | exact joint top-K rankはstable macro-mode IDとして使えない | persistent episode内top-K datum span median 0 / p90 0.0010 ft、70.69%でepisode内path完全一致。best-of-5 / row-oracleはtop-1 RMSE 19.8422を19.8413へしか改善せず、truth bracket率0.0116% |
| 強く確立 | individual global path IDはposterior basin massを運ばない | top-1 joint path probabilityはwell中央値約`10^-466`、top-5合計約`10^-465`。5本はほぼ同scoreのmicro-pathで、macro basin massには桁違いに不足 |
| 強く確立 | global Viterbiのlatent rate stateは元から極端にsticky | 全773 wellsのswitch count median0 / p90 2、75.03%でzero-switch。persistent wellsも71.56%がzero-switch |
| 反証 | latent rate-mode IDのswitch自体がpersistent offsetを必要条件として作る | persistent episodes 67.40% / SSE73.29%はzero-switch well上。switch率とRMSE / mean rate-error増分 / transition crescendoのrhoは`-0.0298 / -0.0410 / -0.0068` |
| 反証 | joint top-Kでrate-mode IDを複数保持すればmacro offsetを回収できる | top-5 rate hashが1種類のwell上にpersistent episodes 68.50% / SSE77.26%。最大prior+truth-GR群では群SSE79.68%。row-level rate path本体は未保存 |
| 反証 | genericな全path diffusenessがoffset severityを決める | top-1 surprisal rate medianはpersistent / nonpersistent=`0.21851 / 0.22176 nats/row`。persistent wells内でRMSEとのrho`0.0047`、persistent row率と`-0.0246` |
| 反証 | posterior meanが二mode間のvalleyを読むことだけが主因 | marginal MAPはepisode平均99.53%でmeanと同符号、5 ft以内はSSE0.41%。ただしrow-wise MAP列は53.92% episodesでhard違反し合法joint pathではないため、mode carrierには使えない。合法なglobal Viterbiでも多くは同方向offset |
| actual messageで確立 | transition/history massが最大の主因 | exp408排他的forward群は452 / 638 episodes、SSE59.40%。重複条件では469 episodes、SSE65.78%。predictive truth-odds strong wrongはrows70.35%、SSE69.15% |
| 強く確立 | confident wrong-GR basinとbroad prior-vs-truth conflictは別regime | prior-opposed+candidate-GRのepisode std 1.46 ftに対し、最大prior-aligned+truth-GR群は7.05 ft。後者はfar→nearで4.07→5.39 ft、79.25% episodes / SSE81.80%が発症前にbroadening |
| severity原因として弱い | smoothed posterior broadening単独 | 全体でfar→near std 3.16→3.96 ftだが、変化量とRMSE rho 0.0919、transition crescendo 0.0414。regime markerでありalpha/betaの起源も未分離 |
| 制御系で反証 | backward smoothing単独が一般にoffsetを作る | centered emissionの制御HMMではcurrent filtered 0.4507をsmoothed 0.0575へ修復 |
| severity modifier | rate span不足 | 10%以上span外の18 episodesはRMSE37.05だが全episode SSEの6.39% |
| severity modifier | missing GR / reliability heterogeneity | high-missing寄与23.64%、reliability AUC0.6076。ただしneutral化・一律weight変更は悪化 |
| 反証 | 固定TVT band / typewell範囲 / MD clamp / 可変rate span / cap / malformed typewell | edge mass・範囲外・clamp発火・cap発火・typewell異常が実質0。spanとepisode RMSE rho -0.056 |
| 反証 | 初期position/rate priorとrate内部遷移の離散化 | position prior bias最大6.94e-17 ft、rate prior bias最大4.00e-7、内部mean誤差最大3.47e-18、floor/cap 0 |
| 反証 | cache merge / replay /並列化・float32 message scaleの数値事故 | exp205 SHA一致、exp270 replay差0、row identity一致。worst-scale float32 stressのsmoothed mean差最大1.48e-4 ft |
| actual messageで確立 | 実prefixからのforward priorがactual offset方向へ関与 | 旧proxyはepisode rho 0.6128、符号一致75.39%。exp408 actual predictive messageではstrong wrongがrows70.35%、SSE69.15%、exclusive forwardがSSE59.40% |
| 強く確立 | position shrinkageの役割はrate basinで反転 | prior-aligned 85.32% SSE群では効果rho -0.7476 / SSE符号一致4.92%、prior-opposed 14.68%群ではrho +0.7243 / 同99.83% |
| actual messageで確立 | predictive prior・filtered alpha・backward beta・path multiplicityの個別寄与 | exp408 exclusive SSE比はforward59.40%、backward23.04%、multiplicity9.04%、support6.39%、mixed2.12%、GR/imputation 0%。multiplicityは重複SSE72.09%の増幅器 |
| 未確定 | position transition変更によるactual-HMM因果効果 | sigma拡大はactual motion分散を11.73xにする。width-onlyとexact-mean transportのactual redecodeはともに未実行 |

したがって原因の骨格は、prefix rate prior、sticky rate dynamics、coarse-grid transition、
GR alias / 相関evidenceと非凸landscape、forward/backwardのjoint basin massが相互作用する
系まで絞れた。
exp408により順序はforward rate prior / transition hysteresisが主因、backward reversalが
第二、sum-product path multiplicityが重複増幅器、state support不足が少数のseverity
modifierと確定した。position量子化はoffsetを作れる条件付き機構だが、actual prefixでは
rate外挿を抑える側にも働く。GR matchingは一部のseed / lock条件だが、current-row
wrong-GRが全体の主因ではない。次の介入を検討するなら、position sigmaやexact-meanを
先に変えるのではなく、rate changeへの追従遅れを抑えつつ正しいsticky区間を壊さない
単一のtransition / reset仮説を、保存済みmessageで対象区間を固定してから設計する。

## 現時点で残る因果分解

position kernel量子化とmean-reversionが、正しいrate basinではoffsetを作るforward
mechanismであることは制御合成HMMで因果的に確認できた。一方、実prefixのtransition-only
priorでは補正効果がactual offsetと逆向きであり、actual filtered rate distributionを
固定しない局所counterfactualからoffsetの向きを決めることはできない。truth pathのhard
support不足は全体原因から除外できた。soft two-step transition NLLがonsetへ向けて上昇する
ため、sticky/coarse grammarによるrate lagは形成トリガーとして強まったが、これは
truth-lateの局所proxyであってactual alpha内のrate massではない。保存artifactだけでは、
actual truth-strongでHMMが誤る群を次の3つへ完全分離できない。

保存trajectoryのrate分解により、直前のabsolute rate errorがfarの`0.01581`から
nearの`0.03365`へ`2.28x`増える一方、真のrate accelerationは増えないことまで確認した。
特に最大のprior+truth-GR群ではtrue `|rate|`が増えるのにdecoded `|rate|`が減り、
zero-directedなunder-responseがdatum差を積分する経路が支配的である。これに対して
prior-opposed+candidate-GR群はtrue `|rate|`低下中にdecoded `|rate|`が増えるため、
wrong-GR駆動の別経路である。ただしsmoothed posterior meanの差分からは、前者を
position shrinkage、mean-reversion、sticky rate transition、backward betaへ分配できない。
合法なglobal Viterbiにもrate-error crescendoが`1.65x`残るため、sum-product平均化だけを
rootとする説明は除外できる。一方、Viterbiは直前16行のSSE加重`66.34%`でmeanより改善する
ので、path multiplicity / posterior meanは重い形成期の増幅器として残る。max-productと
sum-productに共通するtransition/history basinの中身をalpha、predictive rate mass、
position transportへ分けるために後続exp408で実messageを保存した。
行単位の方向分解では、最大prior+truth-GR群のnear rate-error massの`64.66%`が
zero-directed under-response、opposed-prior+candidate-GR群の`96.52%`がovershootだった。
形成経路の異質性までは確定したが、前者の残り`27.21%`を占めるopposite-direction誤差や、
ambiguous / neither群のmixed routeをhidden rate posteriorなしでさらに分けることはできない。

1. forward filteringの時点でtransition priorがtruth basinを落としている。
2. forwardではtruth basinにいるが、future GRを使うbackward smoothingがwrong basinへ戻す。
3. 各方向では真値pathが残るが、sum-productのpath multiplicityがwrong basinの総質量を増やす。

また、Viterbiも改善しない全SSEの15.79%では、coarse-grid position mean bias、
`mom=0.998`、rate-transition boundaryの非再正規化、構造化GR / path multiplicityの
どの相互作用がseverityを決めるかを分離できていない。position width、
exact-mean transport、momentumの独立なactual-HMM介入は未実行であり、message診断より先に
補正variantを走らせる科学的根拠は現時点では弱い。

## 最小診断の実行結果

上記の未観測量はexp408で保存・監査済みである。

- emission適用前のpredictive position mass
- emission適用後のfiltered posterior
- backward適用後のsmoothed posterior
- truth basin / posterior-mean basin / Viterbi basinの各mass
- GR emissionが各basinのlog-oddsを動かした量
- backward messageが各basinのlog-oddsを動かした量
- predictive / filtered / smoothed rate平均・分散・truth/candidate近傍mass
- rate state edge mass、position edge mass、初回escape、再捕捉時刻
- 各行のposition-kernel期待変位・分散、量子化bias、補償rate-stateまでの距離
- 同一のactual filtered source-rate massを固定したcurrent / exact-mean 1-step期待変位差
- position-rate covarianceと、各basin内のconditional rate mean

これにより各persistent episodeを、

- raw-GR alias
- imputation alias
- forward transition/prior hysteresis
- backward smoothing reversal
- sum-product path-multiplicity / posterior-mean readout
- state support不足

へ排他的分類した。結果はforward 452、backward 86、multiplicity 37、support 18、
mixed 45、raw-GR / imputation 0 episodesである。既存exp270に内部messageがなかったという
artifact censusは正しかったため、exp408はparent controlを含む複数variantを回さず、
`1 current variant × 450 HMM well-runs`だけを再decodeした。

対象は`2,264,135 rows`、全suffixの`59.8346%`で、実測は予測約3時間を上回る
`4.425 h`だったがKaggle CPUで完走した。full alpha / beta tensorは保存せず、
638 episode / 807,710 rowsのstream ledger、cause summary、episode summary、
well manifest、runtime metricsとSHAを保存した。truthはmessage計算後のlate diagnostic
maskにだけ使い、HMM入力・state・emission・prediction生成には使っていない。

この結果、message寄与の確定を目的としたStage Aは完了した。次に必要なのは追加の
原因readoutではなく、介入する場合だけ、主因へ直接対応するrate-transition追従仮説を
1つに固定し、保存済みexp209 controlを再学習・再decodeしない差分実験として設計することである。

## 成果物

- `studies/hmm_exp209_grid_boundary_audit.py`
- `studies/hmm_exp209_grid_boundary_audit_20260725/`
- `studies/hmm_exp209_offset_cause_readout.py`
- `studies/hmm_exp209_offset_cause_readout_20260725/`
- `studies/hmm_exp209_transition_kernel_audit.py`
- `studies/hmm_exp209_transition_kernel_audit_20260725/`
- `studies/hmm_exp355_quantization_treatment_readout.py`
- `studies/hmm_exp355_quantization_treatment_readout_20260725/`
- `studies/hmm_exp209_synthetic_transition_mechanism.py`
- `studies/hmm_exp209_synthetic_transition_mechanism_20260726/`
- `studies/hmm_exp209_offset_definition_sensitivity.py`
- `studies/hmm_exp209_offset_definition_sensitivity_20260726/`
- `studies/hmm_exp209_gr_evidence_threshold_sensitivity.py`
- `studies/hmm_exp209_gr_evidence_threshold_sensitivity_20260726/`
- `studies/hmm_exp209_position_resolution_tradeoff.py`
- `studies/hmm_exp209_position_resolution_tradeoff_20260726/`
- `studies/hmm_exp209_float32_message_precision_audit.py`
- `studies/hmm_exp209_float32_message_precision_audit_20260726/`
- `studies/hmm_exp209_existing_transition_intervention_census.py`
- `studies/hmm_exp209_existing_transition_intervention_census_20260726/`
- `studies/hmm_exp209_actual_geometry_transition_prior.py`
- `studies/hmm_exp209_actual_geometry_transition_prior_20260726/`
- `studies/hmm_exp209_prior_emission_interaction_readout.py`
- `studies/hmm_exp209_prior_emission_interaction_readout_20260726/`
- `studies/hmm_exp209_gr_rigid_shift_barrier.py`
- `studies/hmm_exp209_gr_rigid_shift_barrier_20260726/`
- `studies/hmm_exp209_truth_path_grammar_audit.py`
- `studies/hmm_exp209_truth_path_grammar_audit_20260726/`
- `studies/hmm_exp209_truth_grammar_temporal_readout.py`
- `studies/hmm_exp209_truth_grammar_temporal_readout_20260726/`
- `studies/hmm_exp209_onset_transition_gr_timing.py`
- `studies/hmm_exp209_onset_transition_gr_timing_20260726/`
- `studies/hmm_exp209_topk_mode_basin_audit.py`
- `studies/hmm_exp209_topk_mode_basin_audit_20260726/`
- `studies/hmm_exp209_posterior_geometry_timing.py`
- `studies/hmm_exp209_posterior_geometry_timing_20260726/`
- `studies/hmm_exp209_rate_lag_timing.py`
- `studies/hmm_exp209_rate_lag_timing_20260726/`
- `studies/hmm_exp209_decoder_rate_lag_timing.py`
- `studies/hmm_exp209_decoder_rate_lag_timing_20260726/`
- `studies/hmm_exp209_rate_directional_underresponse.py`
- `studies/hmm_exp209_rate_directional_underresponse_20260726/`
- `studies/hmm_exp209_onset_missingness_timing.py`
- `studies/hmm_exp209_onset_missingness_timing_20260726/`
- `studies/hmm_exp209_viterbi_rate_state_stickiness.py`
- `studies/hmm_exp209_viterbi_rate_state_stickiness_20260726/`
- `experiments/exp408_hmm_message_rate_basin_audit/`
- `experiments/exp408_hmm_message_rate_basin_audit/artifacts/readout_v3/`
