# 要件

## 依頼

`public_notebook_catchup_after_self_improvements` を実装する。

exp026 で自前 route の Public LB anchor が 12.102 まで更新されたため、公開上位 notebook を追う段階に入る。ただし、いきなり自前 CV や提出候補へ混ぜず、まず公開 notebook の再取得、依存 artifact、kernel metadata、見えない test で使える リスク、replay 優先順位を棚卸しできる状態にする。

## 制約

- Kaggle Notebook replay は別実験として切る。今回の実装は catch-up inventory と handoff まで。
- 公開 score は Kaggle listing から取れないため、title や notebook 内の明示値だけを記録する。
- formation / Geology / public visible branch / static submission blend は 見えない test で使える 境界の確認対象として扱う。
- replay output が確認できるまで、自前 CV/LB anchor と公開 route を blend しない。

## 受け入れ基準

- 既存または再取得済みの `kernel_listing.csv` と `kernel-metadata.json` から、公開 notebook の依存、手法 family、risk flag、replay priority を生成できる。
- Markdown report と CSV inventory を `docs/notebooks/rogii-wellbore-geology-prediction/` に保存できる。
- 次に切る replay 実験候補と、artifact-stack 系の保留理由が明記される。
- スクリプトの静的検証が通る。
