# 設計

## アプローチ

outer-trainで `horizontal_gr - typewell_gr` を計算し、`group × typewell_gr_decile × abs_gradient_tertile × missing_flag` ごとにweighted medianとMADを推定する。df=5のStudent-t、support `k=200` で上位階層へ縮約し、group conditional→group unconditional→global conditional→global unconditionalの順でfallbackする。評価bankはexp293の固定deployable12とする。各候補TVTでType Well GRを補間してlog likelihoodを作り、候補生成やdecodeをせずtruth-nearest候補の順位だけを評価する。

baselineはouter-train global-unconditional Student-t。group-label shuffleはwell単位でgroup label multisetをSHA256 deterministic rotationし、TVT-shift controlは候補TVT系列をwell内でdeterministic circular shiftする。いずれもouter-valid truthを読まずに候補順位を凍結する。

## 実験範囲

- 対象: `exp312_typewell_group_conditional_gr_emission_table`
- Route: `pf_beam`
- 親: `exp311_typewell_group_prefix_suffix_gr_calibration_readout`
- 変更: scalar residual modelからconditional Student-t tableへ変更。
- 固定: exp293 deployable12 candidate bank、exp311 fold/群定義、df、bin、shrinkage、fallback、truth-late join。
- 計算量: scientific 1 + controls 2、5 folds、model/booster/decoder 0。

## 再現性設計

- bin edgeはouter-trainだけで決定し、stable well順とdeterministic reducerを使う。
- candidate manifest、bin edge、table/fallback/schema/content SHA、rank readout SHAを保存する。
- Kaggle CPU/internet disabled、kernel version、bootstrap config一致を記録する。

## リスクと停止条件

- sparse cell過学習をsupport shrinkageと固定fallbackで抑える。
- exp231のlongtail/worst-well悪化を踏まえ、hidden-like非悪化も必須とする。
- gate FAIL時はbin/df/kを救済せずbranchを閉じる。
- exp311 worst-well FAILを平均gainで無効化したとは扱わず、本実験でもhidden-likeとfold安定性を必須にする。
