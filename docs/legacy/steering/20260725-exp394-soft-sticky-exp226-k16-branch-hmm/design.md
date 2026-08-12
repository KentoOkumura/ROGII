# exp394 設計

## 結論

低ランク 3D 地層場を前提にせず、exp226 の geometry-only path と、exp226/K16 の
相対 rate schedule を持つ free exact HMM を 1 個の switching state-space model として
同時に forward-backward する。別々の予測を後から blend する設計ではない。

## 記号

- `t0`: 既知 TVT prefix の最終 row
- `q^E_t`: group-safe exp226 `tvt_geop`。GR correction / U projection 前の値
- `v_i`: exp209 と同じ absolute-TVT grid
- `rho_j`: exp209 と同じ 41 residual-rate grid
- `c_t in {E,H}`: exp226 branch / free-HMM branch
- `dMD_t`, `dZ_t`: row 間の MD と Z の差
- `r_prefix`: exp209 と同じ既知 prefix tail から得る初期 rate

## K16 relative rate schedule

exp355 と同じ K16 segmentation と reducer を使い、geometry-only rate を

```text
g_t = (Delta q^E_t + Delta Z_t) / Delta MD_t
```

で求める。segment `k(t)` の有限 step の median を `g_k` とし、

```text
mu_t = r_prefix + g_k(t) - g_1
```

へ再 anchor する。先頭 segment が不正なら well 全体、後続 segment が不正なら
その segment だけ `r_prefix` を使う。`mu_t` は HMM 遷移の平均であり、
target GR をまだ使っていない GR 補正前の値である。

## H branch

H branch の state は

```text
x_t = (v_i, rho_j)
```

で、per-well の全 absolute-TVT grid と全 41 rate states を保持する。
HMM の複数 mode はこの grid 上の posterior として保持し、有限個の path へ変換しない。

位置遷移の中心は

```text
m_t(x_{t-1}, rho_t)
  = v_{t-1} - Delta Z_t + (mu_t + rho_t) Delta MD_t
```

とする。position diffusion、residual-rate diffusion、momentum、start prior、
band、grid step は exp209 / exp355 Stage 1 と同じ値を固定する。

## GR 観測モデル

既知 prefix だけから exp209 と同じ affine calibration `(a,b)` と
`sigma_GR` を求める。typewell interpolation を `G_type(v)` とすると、

```text
log e_H(t, v)
  = -0.5 * ((GR_t - (a G_type(v) + b)) / sigma_GR)^2
```

を H branch の全 `v` に評価する。E branch ではユーザーが指定した候補代入を

```text
log e_E(t)
  = log p(GR_t | TVT_t = q^E_t)
```

として同じ式を `v=q^E_t` で 1 回だけ評価する。missing GR の扱い、
interpolation、likelihood cap は exp209 を固定する。

`q^E_t=tvt_geop` は GR を使わず生成されるため、同じ target GR を exp226 correction と
emission の両方に使う二重利用は起こらない。

## soft-sticky regime transition

MD-aware base switching hazard を

```text
h_t = 1 - exp(-Delta MD_t / 1000)
```

とする。E branchの事前平均滞在長は1000 MD-ftで、H branchは後述のdockingにより
1000 MD-ft以上になり得る。初期 prior は `P(E)=P(H)=0.5`。E branch は 1 state、H branch は
全 `(v,rho)` state を持つ 1 個の trellis として log-space exact
forward-backward を行う。

### E からの遷移

- `E -> E`: 確率 `1-h_t` で次の固定値 `q^E_t` へ進む。
- `E -> H`: 総確率 `h_t`。H transition kernel を
  `v_{t-1}=q^E_{t-1}, rho_{t-1}=0` から適用して全 H state へ正規化配分する。

### H からの遷移

正規化済み H transition kernel を `K_H(x_t | x_{t-1})` とする。
この kernel が次 row に提案する全 H state と exp226 path の近さを

```text
d_t(x_{t-1})
  = sum_(v_t,rho_t) K_H((v_t,rho_t) | x_{t-1})
      exp(-0.5 * ((v_t - q^E_t) / 6.0)^2)

P(H -> E | x_{t-1}) = h_t d_t(x_{t-1})
```

とする。残りの確率 `1-h_t d_t(x_{t-1})` を正規化済み H transition kernel に配る。
これにより、遠い H mode から exp226 path へ不連続に teleport する遷移は抑える。

各 source state の outgoing probability は必ず 1 に正規化する。`1000 MD-ft` と
`6.0 ft` は本実験の単一固定値で、full OOF を見た再調整は行わない。

## posterior と出力

forward-backward 後の regime posterior を `gamma^E_t`、H joint posterior を
`gamma^H_t(v,rho)` とすると、最終出力は

```text
TVT_hat_t
  = gamma^E_t q^E_t
    + sum_(v,rho) gamma^H_t(v,rho) v
```

とする。これは branch posterior mean と HMM posterior mean を別々に作ってから
任意 weight で blend する式ではなく、同じ正規化定数を持つ joint posterior の期待値である。

保存する診断は `gamma_E`、`gamma_H`、H conditional mean/std、joint mean/std、
expected switch probability、docking probability、emission log-likelihood、
K16 `mu_t`、fallback flag とする。

## preflight

実装後、最初に fold と suffix length だけで固定した 16 wells
（各 fold の longest 3 wells + 重複しない global median-length well）を処理する。
truth、error、hidden-like role は選択にも gate にも使わない。

必須 technical checks:

- 16/16 wells 完走、finite prediction coverage 1.0
- forward / backward / smoothed posterior normalization error `<=1e-8`
- H full-grid coverage 1.0、branch posterior finite
- source-state transition row-sum error `<=1e-10`
- expected full runtime `<=30,600 sec`
- projected peak RSS `<=25 GB`
- fixed input identity / SHA / leakage guard PASS

preflight RMSE は full run を止める gate にしない。resource gate FAIL は科学的な
negative result ではなく、同じモデルを Kaggle 制限内で exact に実行できない
technical blocker としてユーザーへ戻す。

## full OOF と promotion gate

別承認後に 1 scientific variant、773 switching-HMM well runs、5 reporting foldsを
実行する。保存済み exp263 `8.238331546 ft` を主 baseline とし、exp226
`9.427109596 ft`、exp209 `11.938287235 ft`、exp355 `11.291976616 ft` も報告する。

promotion は次を全て要求する。

- exp263 比 RMSE gain `>=0.25 ft`
- 改善 fold `>=4/5`
- 1000+ ft、hidden-like spatial、hidden-like typewell-purged の regression `<=0.02 ft`
- near 0--250 ft の regression `<=0.02 ft`
- 改善または同等 well 比率 `>0.50`
- paired by-well RMSE delta p95 `<=0`
- worst-well regression `<=0.25 ft`
- global posterior branch occupancy が E/H とも `[0.05,0.95]`
- posterior expected switch rate が `>0` かつ `<=2.0 / 1000 MD-ft`
- exp281 と同じ定義の persistent-offset episode count が exp263 以下
- 512 rows 以内 recovery rate が exp263 以上

gate FAIL 後の同一 OOF parameter rescue、blend、selector、inference、submissionは行わない。

## 過去結果との関係

- exp226 は geometry path の direct signalを示したが OOF `9.4271` で単独 ceilingがある。
- exp355 は K16 relative rate により exp209 を `0.6463 ft`、5/5 folds改善した一方、
  hidden-like と worst well `+52.74 ft` が悪化した。
- exp281/357 は exp226 moving-coordinate residual HMM の平均改善または recoveryを
  一部示したが exp226 direct / exp263 と tail safety を超えなかった。
- exp270 は MAP/Viterbi/top-K path が posterior meanより悪く、top-K追加 oracle価値も
  最大 `0.000342 ft` だった。このため H branch の多峰性は full grid で保持する。
- exp263 は固定物理 blend OOF `8.238331546` / Public LB `7.800` であり、
  本候補の採用 baseline とする。

これらは 2 branch の相補性を試す根拠にはなるが、Public LB 6.5 の実現可能性を
直接示す証拠ではない。

## 再現性

- RNG なし、well / row / grid / rate / branch の順序を固定する。
- CPU、固定 thread 数、log-space deterministic reduction を使う。
- raw identity、decompressed input SHA、schedule logical SHA、branch posterior content SHA、
  prediction content SHA、config / scientific-contract SHA を保存する。
- suffix truth と hidden-like role の read count が prediction freeze 前に 0 であることを
  hard guard にする。

## 実装状態

後続の明示指示により、Jupytext percent形式のcompact self-contained train source、
別名Notebook候補、dense全列挙parityを含む専用test、16-well preflightとfull OOFの
fail-closed orchestrationを実装した。正規train Notebookは既存placeholderを維持し、
Kaggle package、push、run、inference、submissionは作成・実行していない。

## 2026-07-25 fixed16実行承認

後続のユーザー指示により、compact self-contained候補を正規train Notebookへ採用し、
private / CPU / internet-offのcanonical Kaggle packageを作成して固定16-well
technical preflightだけを実行する。technical candidate 1 / HMM well runs 16 /
LightGBM config・trained fold・booster・control rerun・GPU各0。RMSEは計算しない。
full OOF、inference、submissionは承認外のままfail closedする。

## 2026-07-25 fixed16実行結果

canonical private CPU version 1（id_no `128536142`）はfixed16を
`3703.079064 sec`で完了した。数値安定性、全state coverage、identity/leakage、
peak RSSはPASSしたが、773-well wall-clock projectionは`112,736.889439 sec`
（`31.3158 h`）で上限`30,600 sec`の`3.684212x`となった。
事前固定gateに従い`technical_blocker_not_scientific_negative_result`として閉じ、
full OOFへ進めない。再訪はモデル・grid・gateを変えない数値同値な計算最適化が、
同じfixed16上で最低`3.684212x`を先に実証できる場合だけ別設計とする。
