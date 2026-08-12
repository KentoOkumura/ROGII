# 設計

## 仮説

exp226の`K=16`はdirect OOF RMSE `9.4271096741`で提出候補にはならなかったが、exp300では
`>3 ft`悪化wellのselection-regret SSEの52.3%が「oracleはK16なのに別候補へ誤rankingしたrow」に集中した。
Kは坑井pathを等分する幾何segment解像度なので、粗い`K=12`または細かい`K=24`へ1変数だけ動かすと、
direct errorまたは既存物理bankにない局所誤差構造が改善する可能性がある。

ただしexp298ではexp226局所形状のprimary componentがH256/H512ともrank 4/5・5/5で弱かった。
したがって広いK gridは正当化せず、K16の両側1点だけで反証可能な監査に限定する。

## 実験範囲

- 対象実験: `exp302_exp226_multiscale_k_segment_candidate_audit`
- Route: `pf_beam`
- 親実験: `exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`
- 比較bank: exp293 fixed deployable12
- 変更する変数: `model.params.k_segments`のみ (`12`, `24`)
- 固定する変数: exp226の全残余設定、fold、row、scope、block、tie、閾値、入力契約
- 対象外: HMM、候補weight、selector、decoder、inference、submission

## variant

| variant | `k_segments` | 役割 |
| --- | ---: | --- |
| saved control | 16 | 保存済みOOFを読む。再生成しない |
| `exp226_k12` | 12 | 粗いsegmentによるvariance低減仮説 |
| `exp226_k24` | 24 | 細かいsegmentによる局所追従仮説 |

`K=8/20/28/32`、Kの連続探索、fold別K選択は行わない。

## 評価

### Direct readout

K16 controlと各variantを、pooled/fold/distance bucket/1000+/hidden-like spatial/
hidden-like typewell-purged/by-well p50・p95・worstで比較する。

少なくとも1 variantが次をすべて満たした場合だけdirect PASSとする。

1. pooled RMSE `<= 9.3771096741`（K16から`>=0.05 ft`改善）。
2. 4/5 foldsでK16より改善。
3. 1000+、hidden-like 2面のdeltaがそれぞれ`<=+0.02 ft`。
4. by-well RMSE p95とworstのdeltaがそれぞれ`<=+0.25 ft`。

### Candidate novelty readout

exp293 fixed deployable12にK12またはK24を1候補ずつ別々に追加する。K12とK24を同時追加した値は
secondary readoutにも使わず、各candidate固有の増分だけを測る。exp293のblock origin、非重複
H128/H256/H512、final short block、SSE objective、tie tolerance/orderを固定する。

少なくとも1 variantが次をすべて満たした場合だけcandidate novelty PASSとする。

1. H512 oracle RMSEがfixed12から`>=0.03 ft`改善。
2. whole-well oracle RMSEが`>=0.02 ft`改善。
3. H512 strict unique-best block比率が`>=2%`。
4. H512 oracle RMSEが4/5 foldsで改善。

row/H128/H256のoracle、candidate correlation、unique-best、well別選択率はsecondary診断として記録するが、
primary guardを差し替えない。oracle predictionは保存しない。

## 判断規則

- direct PASS: direct candidateとして別途inference設計を検討できるが、このexpでは実行しない。
- candidate novelty PASS: exp303の科学的先行条件を満たす。
- direct PASSかcandidate novelty PASSのいずれか: 仮説にscientific supportあり。
- 両方FAIL: branchを閉じ、K値・kappa・GR・gate・smoothnessの同一OOF救済を追加しない。

## freeze順序とリーク防止

1. exp226 saved OOFから`well_id,row_idx,suffix_offset,tvt_pred,fold`だけをallowlistで読む。
2. exp226 decompressed content SHA、raw input SHA、fold/well/row identityを照合する。
3. K12/K24をouter-valid well完全除外で生成し、finite/row/order/control parityを確認する。
4. K12/K24 prediction content、exp293 bank、block assignment、scope、tie、threshold manifestをSHA freezeする。
5. 別loaderでtrue suffix TVTを接続し、directとoracle readoutだけを行う。

K16 control parityは保存済み値との差最大絶対値`<=0.001 ft`を要求する。truth-before-freeze accessは0でなければ停止する。

## 再現性設計

- seed policy: exp226のdeterministic sortingとSHA256 fold assignmentを再利用する。
- stochastic処理: なし。
- PF/Beam乱数: なし。routeは物理候補分類による。
- 並列処理: 初回は`num_workers=1`。global RNGを使用しない。
- runtime: Kaggle CPU、internet disabled。GPU/boosterなし。
- SHA: raw input、K16 OOF decompressed content、exp293 bank/block、feature schema、K12/K24 prediction content、
  fold/subgroup/readoutを記録する。
- Kaggle bootstrap: 実装・pushが別途承認された時点でcanonical configとbootstrap内configを照合する。
- model/submission SHA: modelとsubmissionを生成しないため対象外。

## リスク

- リーク: donor fieldまたはadaptive kappaへouter-valid wellが入ると無効。exp226のwhole-well除外を固定する。
- CV/LB: K16はCV 9.427 / Public LB 9.837で不採用。小幅direct改善だけで提出価値を主張しない。
- 多重比較: K12/K24の2点だけでもbest-of-two biasがあるため、fold/longtail/well guardを全必須にする。
- ランタイム: 2 variants × 5 folds。K16再生成を禁止しCPUコストを抑える。
- 再現性: gzip byte SHAではなくdecompressed content SHAを主証拠にする。
