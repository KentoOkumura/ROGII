# 要件

> **閉鎖済み（2026-07-22）**: 親exp323のterminal closeにより未実装・未実行で閉鎖した。新相当実験はexp338と新exp323相当の二段階PASS後に新番号でのみ作成する。

## 依頼

exp226 window likelihoodを時間変化する`lambda_t`としてexact HMM内部へ移植する設計を確定する。実装しない。

## 制約

- Route: `pf_beam`。exp305結果確認とexp323 promotionが必須。
- 補正TVT、exp226 prediction、peer atlas、prediction blendを作らない。
- window/stride/score weight/lambda式は1本に固定しgridを禁止する。
- score surfaceとlambdaをtruth-freeに凍結する。
- Stage 0 FAILならHMMへ進まない。

## 受け入れ基準

- exp321との差、二重計上対策、lambda式、coverage/fallbackが明記される。
- Stage 0/1 gateと最大実行量が一致する。
- 実装/Kaggle/inference/submissionが無効である。
