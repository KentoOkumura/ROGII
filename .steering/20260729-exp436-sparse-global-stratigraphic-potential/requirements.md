# 要件

## 依頼

物理モデルの状態遷移を改善する第2案として、outer-train wellsから地層面別の
滑らかな物理ポテンシャル

```text
U_k(X, Y) = TVT_contact,k + Z_contact,k
```

を疎な大域問題として推定し、その空間差で未知suffixを進める
`exp436_sparse_global_stratigraphic_potential`を設計する。

初回設計セッションでは既存の誤った「単一`P(X,Y)`」設計を訂正し、バックログ、
steering 3文書、実験ディレクトリの設定・記録文書までを同じ契約へ揃えた。
2026-07-29の追加ユーザー指示`exp436を実装してください`により、compact
self-contained train候補とcontract testの実装だけが承認された。正規notebook採用、
Kaggle package/push/run、推論、提出は引き続き行わない。

## 仮説

exp226は局所donor rateを逐次積分するため、小さなdonor mismatchが長距離offsetになる。
一方、単一`P(X,Y)`は同じ水平位置に複数の地層準位が存在する物理を潰してしまう。
6つのformation contact
`ANCC / ASTNU / ASTNL / EGFDU / EGFDL / BUDA`ごとに、outer-train全wellの
contact `U=TVT+Z`を1枚の正則化面として同時推定すれば、局所donor転写を使わず、
地層準位を保った保存的な遷移を作れる。

各formation `k`のtarget pathは最後の既知点`a`へanchorし、

```text
delta_U_k(t) = U_hat_k(X_t, Y_t) - U_hat_k(X_a, Y_a)
TVT_hat_k(t) = TVT_input(a) + delta_U_k(t) - (Z_t - Z_a)
```

とする。primary candidateはanchor時点でsupportを満たす固定formation集合の
`delta_U_k`を等重み平均した1つの保存場差であり、suffix途中のformation切替、
target生formation列、selectorを使わない。

## 制約

- Routeは`pf_beam`。
- 親/controlは`exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction`。
- exp226 controlは保存済みOOFをSHA固定で読み、再生成しない。
- surface observationはouter-train wellだけから作る。
- outer-valid wellの真TVTと6 formation列はsurface fitへ入れない。
- target側で使える列は`well_id/row_idx/MD/X/Y/Z/TVT_input`だけ。
- target/testの生formation列、GR、suffix truthはcandidate freeze前に読まない。
- source contactは`Z-F_k=0`のMD昇順first crossingと線形補間に固定する。
- formationごとに1つのglobal sparse surface/foldを解き、queryごとのlocal plane/
  surface solveを行わない。
- primaryは固定support集合の等重みpotential平均。row-wise k選択、重み学習、
  parameter grid、fallback、HMM/PF/Beam、ML、selector、blendを禁止する。
- exp381のcontact-TVT FAILを覆さない。絶対contact-TVTではなくanchor差だけを検証する。
- 実装承認は取得済み。正規notebook採用とKaggle実行は新しいユーザー承認を必要とする。

## 受け入れ基準

- steering 3文書と実験配下の文書・設定が同じ地層面別実装済み・実行ロック契約を持つ。
- `KAGGLE_DIRECTION.md`の第2案を単一`P`から6つの`U_k`へ訂正する。
- `experiment_summary.md`を更新し、状態を`stage0_implemented_unrun`とする。
- 5 folds × 6 formation surfaces=`30` global fieldsを上限とする。
- Stage 0 resource/integrity、Stage 1 prefix rolling-origin、Stage 2 truth-late OOFの
  順序とfail-close条件を確定する。
- Stage 2はexp226比`0.25 ft`以上改善、4/5 folds、固定scopeとby-well tailのAND gate。
- deterministic anchorは同一設定rerunでlogical content SHAが一致するまで主張しない。
