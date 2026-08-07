# 要件

## 依頼

exp209 exact HMMの持続rate状態を`U-rate=d(TVT+Z)/dMD`ではなく
`TVT-rate=dTVT/dMD`として持つ科学的比較実験を、backlog、steering、
実験ディレクトリへ確定し、段階別承認に従って実装・評価する。

2026-07-30の追加依頼「exp446を実装してください」により、別名のcompact
self-contained train候補、inference guard、専用contract testの実装までを
承認範囲へ追加した。同日の追加依頼「実行してください」により、正規Notebook
採用、Kaggle package作成、固定Stage 0を承認範囲へ追加した。Stage 1、
inference、submissionは引き続き未承認とする。

Assumption: 「その実験」はexp445のような同値な座標再ラベルではなく、
`(TVT, persistent TVT-rate)`へ動力学を変更する新しいtreatmentを指す。

## 根拠

- exp445は`TVT position <-> row-shifted U position`の同値性を検証済みだが、
  rate dynamicsはexp209のU-rateのままであり、本仮説を検証していない。
- exp435はrate履歴を除去したTVT-only HMMであり、persistent TVT-rateを
  状態に持たないため、本仮説のnegative controlではない。
- exp441はU-rateの全support OU化であり、rate座標をTVT-rateへ変えていない。
- exp408はexp209のpersistent offsetでforward transition/prior hysteresisと
  rateの0方向under-responseが大きいことを示している。

## 制約

- Route: `pf_beam`。
- 親は`exp209_exp072_exp205_joint_exact_parity_fast_cache_generation`。
- scientific variantは`persistent_tvt_rate`の1本だけ。
- 状態は`(TVT position, TVT-rate)`。rate定義、prefix rate初期値、
  rate grid、position meanを一貫してTVT-rateへ変更する。
- `n_rates=41`、最小span `0.10`、margin `0.04`、`momentum=0.998`、
  `sig_r=0.002`、`sig_p=0.02`、TVT grid、GR emission、position prior、
  posterior readoutはexp209から固定する。
- Stage 0はfixed32の1候補×32 wells。保存exp209 controlを使用し、
  parent HMMを再実行しない。
- Stage 1はStage 0の全AND gate PASSと別のユーザー承認がある場合だけ
  773 wellsで実行する。
- rate span、momentum、noise、grid、emission、prior、gate、blend、
  selectorをsame-OOFで救済しない。
- ML model、LightGBM、booster、PF、Beam、GPU、inference、submissionは0。
- 再現性は`docs/06_reproducibility.md`に従い、RNGなし、固定順、
  decompressed content SHAを主証拠とする。

## 受け入れ基準

- 親とcandidateのrate定義、初期化、grid、transition、position更新式が
  一意に記載され、単なる座標変換との違いが明確である。
- `Z`がprefixからsuffixまで一定のsynthetic sentinelでは
  `U-rate=TVT-rate`となり、exp209とcandidateが数値一致するcontractがある。
- small-state brute-force、rate/position kernel、posterior normalization、
  truth-late、SHA readbackのtechnical gateが固定されている。
- fixed32 mechanism gateとStage 1 promotion gateが事前固定されている。
- 実行量はStage 0候補32、Stage 1候補773、parent rerun 0、
  model / booster / PF / Beam / GPU 0として全文書で一致する。
- 初回runをdeterministic anchorとせず、独立rerun時だけinput、
  transition、posterior、prediction SHA一致を確認する。
- compact候補と専用testが実装され、constant-Z、small-state dense reference、
  truth-late、SHA readback、全gate key消費のcontract testを通る。
- 正規Notebook、Kaggle package、固定Stage 0だけが有効である。
- Stage 1、inference、submissionが未承認・無効のままである。

## Stage 0実行結果

Kaggle private CPU version 1（id_no `129106260`）でfixed32を完走した。
technicalはruntime projectionだけFAILして`17/18`、mechanismは`0/7`。
matched control pooled / p95が`+7.159063 / +16.310622 ft`悪化し、
under-responseおよびforward/persistent SSEも悪化したため、
`stage0_fail_closed`で終了する。Stage 1、rerun、inference、submissionは
実行しない。
