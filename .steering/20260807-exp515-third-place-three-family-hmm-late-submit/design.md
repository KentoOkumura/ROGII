# 設計

## アプローチ

原writeupの絶対TVT全域gridはstate discretizationが非公開で、bias/referenceを素朴に追加するとKaggle上限を超える。そこでlast-256 rateから得たbaseline pathの周囲`±10 ft`をTVT offset latticeにし、formation rate、bias、reference identityをjoint stateに持つcompact exact HMMとして再構成する。`±10 ft`は原writeupの最終integrated-correction capと整合する。

same-typewell groupは、train typewell GRのnative exact-overlapを調べたexp065 assignmentをtarget-free identityとして使う。train OOFでは各validation foldをatlas sourceから除外し、残り4 foldsのhorizontal `TVT,GR`だけを0.25/0.0625 ft binsへmedian aggregationする。testはruntimeのtest typewellとtrain typewellのnative k-gram一致でgroupを解決し、全train wellsからatlasを作る。

## 公開説明と実装の差

- 実装区分: `proxy`
- 参照sourceとの一致点: 3 families、weights 0.50/0.20/0.30、same-typewell sibling bins、self-prefix reference、TVT/rate/bias/reference hidden state、prefix 128 clamp、rate init 256、Student-t df=1、smooth transitions、exact forward-backward、rate/correction projection。
- 参照sourceからの変更 / 省略点: source非公開のためcompact offset lattice、9 rate states、9 bias states、reference switching probability、self shrinkage、Local-DTW stretchを固定推定する。Local-DTWはstretch-state HMMによる構造再現で、原コードparityではない。
- input tensor / feature: rowごとのMD/Z/GR、known TVT clamp、state TVTで補間したtypewell/sibling/self GR reference。
- target / objective: probabilistic state estimation。truthはdecodeに使わずOOF scoreだけに使う。
- output representation: per-row posterior mean TVTとfamily mixture、posterior std、family log evidence。
- loss: なし。Cauchy emission log likelihood。
- decode / postprocess: exact sum-product forward-backward、固定family convex mix、rate projection。
- context unitと予測範囲: last 128 known + full unknown suffixのwhole-well。
- この実験が支持 / 棄却できる主張: 公開仕様を再構成した3-family HMMがuser exp209系よりOOF/LBで改善するか、reference family多様性が効くか。
- この実験では判断できない主張: 原チームcode parity、非公開パラメータの正しさ、最終5-model ensemble score。
- 実験名と実装機構の整合: `third_place_three_family_hmm_late_submit`は3-family HMMのwriteup再構成かつlate submission auditを表し、exact source reproductionとは表記しない。

## 探索幅とpivot判定

- 変更class: `mechanism` + `representation`
- 同じ親 / familyで連続した小改善実験数: exp209派生のemission/transition小変更は多数あり、単一typewell HMMの限界が反復している。
- positiveなoracle headroom / coverage / 誤差非相関性: 3位writeupではbase 6.0492から3-family 5.9703へ改善し、Local-DTWの誤差非相関性を明示している。
- 比較したrepresentation-change案: user exp209 absolute-TVT/rateのみ、exp223 self-GR、exp231 sibling atlas auxiliary emission、今回のjoint bias/reference + 3-family mixture。
- 小改善の継続またはpivotを選ぶ根拠: 公開された上位解法のreference designへmechanism pivotする明示依頼である。
- `kaggle-idea-forge` の実行要否と根拠: 不要。発想生成ではなく指定済み3位解法の再構成である。

## 実験範囲

- 対象実験: `exp515_third_place_three_family_hmm_late_submit`
- Route: `pf_beam`
- 親実験: scientific sourceは3位writeup、コード構造参照はexp374/exp231/exp065。
- 変更する変数: HMM state、reference construction、prefix conditioning、emission、3-family mixture、projection。
- 固定する変数: 3種類の混合比と、公開されていないため推定した全設定値を実行前に固定し、OOF/LB後に変えない。

## 再現性設計

- seed policy: RNGなし。well/fold/family/state/reference bin順をsort固定する。
- stochastic 処理の有無: なし。
- PF/Beam / likelihood-PF / seed bagging の有無: すべてなし。
- 並列処理と乱数の関係: familyは固定順、well outputはIDでsortしてmergeする。thread schedulingは数値入力やmerge順を変えない。
- CPU/GPU runtime と deterministic flags: Kaggle private CPU、GPU/TPU/internet off。Numba float32 exact lattice。
- train cache / test feature regeneration の SHA 記録方針: group assignment、reference manifest、prediction logical content、runtime schema/row/well数を記録する。
- model manifest / prediction / submission SHA 記録方針: trained modelは0。代わりにreconstruction manifest SHA、OOF content SHA、test prediction content SHA、submission SHAを記録する。
- Kaggle package bootstrap 確認方針: canonical config/source SHAとembedded bootstrapの同一性、CPU metadata、competition/kernel sources、slug/titleをpush前に確認する。

## リスク

- リークリスク: sibling horizontal `TVT`を使うためvalidation fold全wellをsourceから除外する必要がある。prediction freeze前のvalidation truth accessを契約testで禁止する。
- CV/LB 不一致リスク: hidden typewell group mapping、test spatial/typewell composition、非公開原splitが異なる。
- ランタイム/メモリリスク: Local-DTW 3 reference statesでalpha historyが最大。well逐次処理、family逐次解放、compact `41x9x9xF` stateで抑える。
- 再現性リスク: Numba/thread reductionの微小差。state/merge順を固定し、submission content SHAを記録する。
- 手法忠実性リスク: 非公開パラメータとLocal-DTW詳細は一致不能。resultで原CV 5.9703との単純parityを主張しない。
- 過度な縮小 / proxy化リスク: `±10 ft` offset bandがraw posteriorを切る可能性がある。boundary posterior massを記録し、ただし同一run後にbandを拡張して救済しない。
