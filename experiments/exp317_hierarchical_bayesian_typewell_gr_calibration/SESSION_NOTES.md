# exp317 セッションノート

## 現在の状態

- 2026-07-21: 設計確定。exp311/313待ち。実装・実行なし。
- Route: `pf_beam`
- 実行契約: primary 1 + diagnostics 3 / 5 folds / 0 booster / 0 decoder。

## 設計契約

- Student-t df=5、global/group/well hierarchy、deterministic MAP/Laplaceを固定する。
- primaryはsigma-only hierarchy、identity affine。full affineはdiagnosticのみ。
- optimizer収束をtechnical gateとし、silent fallbackしない。
- posterior parameter/model manifest/schema/content SHAを保存する。

## 次

先行gate PASSと実装承認後にdeterministic posterior predictive readoutを作る。
