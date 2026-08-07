# exp314 セッションノート

## 現在の状態

- 2026-07-21: 設計確定。exp311/313 PASSとGPU/CPU学習承認待ち。実装・実行なし。
- Route: `ml_model`
- 実行契約: 1 variant / 3 LightGBM configs / 5 folds / 15 boosters / control 0。

## 設計契約

- 追加はgroup support 2列、sigma、fit RMSE、|bias@GR50|、availabilityの6列。
- exp148 fold/features/configsとsaved OOF controlを固定する。
- outer-valid wellの0件fit assertion、feature/schema/content SHA、15 model SHA、OOF SHAを記録する。
- direct calibrated GR、control再学習、inference、submissionは禁止する。

## 次

先行gate PASS後、15 boostersの明示承認を得てから実装する。
