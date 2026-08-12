# 設計

## アプローチ

重いGR residualがGaussian二乗誤差でalignment modeを固定する仮説を、まずexp280の固定shift bankで反証する。Student-t scoreがrankとstress scopeを改善した場合だけ、exp281 residual-offset HMMのemission familyを1箇所置換する。

## 実験範囲

- 対象実験: `exp342_exp226_student_t_residual_offset_emission_audit`
- Route: `pf_beam`
- Stage 0親: exp280、Stage 1親: exp281。
- 変更する変数: emission family Gaussian→Student-t df=4だけ。
- 固定する変数: base sigma、missing、GR/typewell、exp226 path、shift/block、HMM grammar、transition、prior、posterior mean。
- 実行量: Stage 0は1 scientific score + saved Gaussian control、HMM 0。Stage 1は1 variant / 773 runs、control再実行0。

## 検証方法

1. exp280 inputsとGaussian rankを数値再現する。
2. truthなしでStudent-t block scoreとcontent SHAをfreezeする。
3. truth-nearest shiftをlate joinし、rank/fold/stress/shuffle/extreme-residual gateを判定する。
4. PASSと別承認時だけexp281 grammarでfull HMMを実装する。

Stage 0 implementationではGaussianを再計算せず、SHA固定済みexp280
target-free scoreをcontrolとして読む。Student-tとGaussianに同一のwell/block固有
nonzero circular shift-bank rotationを適用する。truth-nearest shiftで`|z|>=3`が
1行以上あるblockをextreme scopeとし、top3とmean regretの両方を厳密改善させる。

## 再現性設計

- RNGなし。shift/block/well順固定。shuffleはstable SHA256 rotation。
- CPU、GPU/internet off。Stage 1最大8.5時間。
- input/scientific contract/block score/prediction/metricsのdecompressed content SHAを記録する。
- deterministic submission anchorではなく、inference/submissionはdisabled。

## リスク

- score flattening: wrong stateの大残差も弱め、alias識別力を落とす可能性をmargin/rankで先に監査する。
- exp281 negative rescue: full decode前に独立したheavy-tail Stage 0を必須とする。
- multiple testing: df=4の1設定だけを使い、Huberはexp344の条件付き別仮説とする。

## 優先度

高リスク`P3`。Stage 0は安価だがfull HMMはP1/P2後。

## 次のアクション

Stage 0と探索override Stage 1はいずれも固定gate FAILで完了した。
追加救済、再実行、inference、submissionは行わない。

## 2026-07-23 Stage 1実装方針

Stage 0 FAIL後の探索的overrideとして、exp281 compact self-contained exact
forward-backward kernelを同じexp342 train notebookへ取り込む。decoderへ渡すのは
raw horizontal/typewellとSHA固定済みexp226 `tvt_geop`だけとし、Student-t pathを
全773 wellsでfreezeした後に、保存済みexp281 OOFのGaussian parent、exp226、
truth、fold、`md_since`をjoinする。

変更点は行別GR emissionの
`-0.5*min(z^2,600)`から`-2.5*log1p(z^2/4)`への置換だけとする。
parent Gaussian HMMを同一runで再実行しない。保存済みexp281 OOFの
decompressed/content SHAと親RMSEをhard guardし、runtime差が比較へ影響しない
ことを記録する。

Stage 1 scientific gateとdirect promotionを分離する。scientific gateはexp281比、
fold、1000+、hidden-like、well-tail safetyのAND。direct promotionはこれに加えて
exp226 RMSEを更新した場合だけtrueとする。いずれもinference/submissionへ
自動進行しない。

## 2026-07-24 Stage 1設計判定

実HMMではGaussian exp281比`0.047648 ft`の小幅改善を確認したが、事前下限
`0.05 ft`、4/5 folds、hidden-like非劣化、well-tail safetyを満たさなかった。
technical parityとcoverageはPASSしているため、単一変更のStudent-t emissionが
平均では少し効いても、一貫性とtail riskを改善しないnegative resultと解釈する。
