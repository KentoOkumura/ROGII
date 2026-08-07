# exp313 セッションノート

## 現在の状態

- 2026-07-21: 設計確定。exp311/312 artifact待ち。実装・実行なし。
- Route: `pf_beam`
- 実行契約: 3 audit surfaces / 5 folds / 0 model / 0 booster / 0 decoder。

## 設計契約

- exact groupはpeer≥2かつeffective rows≥64だけ利用する。
- fallbackはexact→将来PASS済みsoft→global→identity。現defaultはidentity。
- suffix truth/error、formation train-only、well ID ruleはavailabilityへ使わない。
- membership/availability/fallback/prior source/readout SHAを記録する。

## 次

exp311/312 PASSと実装承認後に共通transfer guardを実装する。
