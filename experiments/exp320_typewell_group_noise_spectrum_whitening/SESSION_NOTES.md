# exp320 セッションノート

## 現在の状態

- 2026-07-21: 設計確定。exp311/313待ち。実装・実行なし。
- Route: `pf_beam`
- 実行契約: primary 1 + diagnostics 2 + control 1 / 5 folds / 0 model / 0 booster / 0 decoder。

## 設計契約

- contiguous finite run、Yule-Walker AR(1)、support≥64、shrinkage k=200、rho clip[-0.8,0.8]。
- fallbackはrho=0 raw likelihood。lag/order/clip/gridは変更しない。
- run/rho/fallback/candidate score/readoutのschema/content SHAを保存する。
- GR smoothing、decoder、inference、submissionは禁止する。

## 次

先行gate PASSと実装承認後にinnovation-rank readoutを実装する。
