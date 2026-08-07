# exp511_exp413_transductive_k16_neighbor_rate_postprocess

## 状態

- route: `ensemble`
- 状態: Kaggle Stage A完了・性能gate FAIL・終端閉鎖
- CV: `7.883964795205812`（exp413 `7.884802794404715`、gain `0.000837999 ft`）
- Public / Private LB: 対象外
- 作成日: 2026-08-04
- 親実験: `exp413_scale5_likpf_full_replacement_on_exp335`

## 実験内容

保存exp413 OOFを固定し、outer-valid foldを擬似test batchとして、他wellの予測TVTとraw
`X/Y/Z`だけからK16 local-linear neighbor-rate consensusを作った。固定1候補へ
`alpha=0.05`、`±0.25 ft` capを適用し、prediction freeze後にだけtruthとhidden-like roleを
接続した。model、booster、PF/HMM/Beam、GPU、親再学習はすべて0。

## 仮説

予測済みTVTの低周波K16 rateについて、同じ擬似test batch内の他wellだけから空間合意を作り、
自wellとの差を弱補正すれば、exp413のpathを壊さずcross-well不整合を減らせる。

## 検証方針

保存exp413 outer 5-fold OOFを固定し、pooled gain `>=0.01 ft`、nonworse `>=4/5 folds`、
fixed scope、by-well tail、first-row continuityを全AND判定する。予測phaseは
`well,row_idx,fold,pred_tvt,X,Y,Z`だけを許可し、freeze後にtruthを読む。

## 結果

- technical checks: 全PASS。
- pooled gain: `0.000837999 ft < 0.01 ft`でFAIL。
- nonworse folds: `2/5 < 4/5`でFAIL。
- scope、by-well tail、continuityはPASS。
- support成立率: `350 / 12,368 = 2.8299%`。
- 最終判定:
  `FAIL_CLOSE_WITHOUT_ALPHA_CLIP_K_BANDWIDTH_RHO_THETA_SUPPORT_FADE_SCOPE_OR_GATE_RESCUE`。

## 実行入口と生成物

- 正規train Notebook:
  `exp511_exp413_transductive_k16_neighbor_rate_postprocess_train.ipynb`
- Jupytext source:
  `exp511_exp413_transductive_k16_neighbor_rate_postprocess_compact_selfcontained_train.py`
- inference Notebook: template placeholderのまま、無効。
- Kaggle kernel:
  `kentookumura/exp511-exp413-k16-neighbor-rate-postprocess-train` version 4。
- 取得済みartifact: `kaggle/output/train_v4/artifacts/`。

## 結論

平均はわずかに改善したが、事前固定したmaterial gainとfold安定性を満たさなかった。同一OOFで
alpha、cap、K、bandwidth、rho、theta、support条件を調整せず、inference / submissionを作らず
閉鎖する。詳細なmetricとSHAは`result.md`、`metrics.json`、`SESSION_NOTES.md`を参照する。

## 所見

support成立率2.83%と低く、選択距離中央値も4,291 ftだったため、predicted-onlyの近傍情報は
強いexp413をfold安定に補正するには疎だった。tail安全性は保てたが、material gainへ届かなかった。
