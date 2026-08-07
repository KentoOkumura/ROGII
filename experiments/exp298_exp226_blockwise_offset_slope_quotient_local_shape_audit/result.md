# exp298_exp226_blockwise_offset_slope_quotient_local_shape_audit 結果

## 仮説

exp226 pre-U pathは大局的offset/slopeを除いたH256/H512局所形状で、exp293 deployable12より強い。

## 設定

- 親: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 比較: exp293 fixed deployable12
- 検証: 保存5 folds、H128/H256/H512非重複block、truth-free freeze後readout
- primary metric: H256/H512 blockwise affine-quotient RMSEとrank
- シード: 42（RNGなし、順序固定用）
- Route: `pf_beam`
- 実装状態: 正規train Notebook採用済み、Kaggle private CPU version 2完了
- Kernel: `kentookumura/exp298-exp226-local-shape-quotient-audit-train` version 2 / id_no `127956072`

## 親からの変更

exp226の候補生成や予測値は変更せず、保存済み中間成分とexp293固定候補に対するblockwise quotient readoutを
実装した。target-free allowlist、candidate/component/block freeze、post-freeze truth loader、固定PASS判定を持つ。
deployable correction、decoder、modelは追加していない。

## 結果

| メトリック | 値 |
| --- | --- |
| CV | - |
| Public LB | - |
| Private LB | - |
| H256 affine-quotient RMSE / rank | 0.3482675905 / 4位 |
| H512 affine-quotient RMSE / rank | 0.7224085771 / 5位 |
| H256 / H512 post-U RMSE | 0.3041197991 / 0.6096467779 |
| H256 / H512 top3 fold数 | 0/5 / 0/5 |
| H512 1000+ / hidden-like 2面 rank | 5 / 5 / 5 |
| technical decision | PASS |
| scientific decision | FAIL / branch close |
| dedicated tests | 11 passed |

## 再現性

- deterministic anchor: false。fixed-input diagnosticのみ
- seed policy: no RNG / fixed fold-well-row-candidate order
- kernel version: 2、id_no `127956072`、status COMPLETE
- candidate bank content SHA: `29477141685662bae7417e788ec5dbe914c2220b7f4cc45ab01befa4e5e3b474`
- component content SHA: `41390811200b2bb5e9876db4beac1eb21cacc73590b5b67849c320c065b1ef10`
- block assignment decompressed SHA: `b0755c22aa8d791012d3f605e2f1b66063ce9bb6ba46ddd4b48dca77cce032d7`
- truth content SHA: `e9067327058431278a0fd994e8e6005b76ab99acbd3942118974599afb69a8d0`
- readout content SHA: `9dec97ae83b83a65c2118a42f217a4cbec499004bdfbb7f3d087cd2178184edd`
- 取得した小規模metrics/manifestはKaggle SHA manifestと全件一致
- model SHA / manifest SHA: modelなし
- prediction SHA: deployable predictionなし
- submission SHA: submissionなし
- rerun result: version 1は監査後の表示`KeyError`。表示キーだけを修正したversion 2で主要値が完全一致し完走

## 解釈

singleton除外、truth-free freeze、bank/block SHA、allowlist、finite/coverage、alias parityを含むtechnical guardは
全PASSした。H128/H256/H512のsingletonは`4/2/2 blocks`を全候補共通で除外し、affine-eligible row coverageは
1.0だった。したがってFAILはsingletonや実装不備ではなくscientific resultである。

入力行数preflightでは最終block長1がH128/H256/H512で`4/2/2 wells`存在した。ユーザー承認により、
exp293のblock境界とSHAは保持し、singletonをaffine RMSE/rank/block win/strict unique-bestの分母から
全候補共通で除外する。technical coverage 1.0は長さ2以上のaffine-eligible rowsへ要求し、singleton数を
生成物へ記録する。長さ2以上のinvalid blockは従来どおりfallbackなしのtechnical FAILとする。

`P_preU`はstrict unique-best block比率だけはH256 `0.113630`、H512 `0.115478`で閾値0.05を通過した。
一方、H256/H512 pooled rankは4/5位、全foldで同じ4/5位、1000+とhidden-like 2面もすべて5位だった。
さらにpost-UよりH256で`+0.044148 ft`、H512で`+0.112762 ft`悪い。したがって、U projection前の
`tvt_geop + gr_delta`を局所sourceとして固定hybridへ持ち込む仮説は支持されない。

oracle offset/slope quotientは局所形状を分離する診断であり、その係数、補正prediction、選択predictionは
保存していない。model、booster、PF/Beam再生成、inference、submissionも生成していない。

## 次

固定契約どおり本枝を閉じる。`downstream_branch_contract.md` Stage 2/3/4、component/horizon/quotient/
scope/平滑化/weightの救済grid、inference、submissionへ進まない。exp293/exp297 fixed12とexp295独立SSMは
変更しない。本結果だけを根拠とする新規救済backlogも追加しない。
