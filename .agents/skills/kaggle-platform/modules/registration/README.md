# Registration — Account & Credential Setup

Kaggleアカウントとcredentialの設定を案内する。詳細な手順と認証方式の正本は[references/kaggle-setup.md](references/kaggle-setup.md)とし、このREADMEへ重複させない。

## 手順

1. credential checkerを実行し、すでに利用可能な方式を確認する。

   ```bash
   uv run python .agents/skills/kaggle-platform/shared/check_all_credentials.py
   ```

2. 実行するclientに合う方式を選ぶ。
   - Kaggle CLIはOAuth、API token、legacy username/keyを使用できる。
   - Kaggle Python APIとkagglehubはAPI tokenまたはlegacy username/keyを使用できる。
   - MCPはKaggle Settingsの「Generate New Token」で生成したAPI tokenを使用する。
3. API tokenをローカルファイルへ保存する場合は、ユーザー自身が対話端末で次を実行する。

   ```bash
   uv run python .agents/skills/kaggle-platform/modules/registration/scripts/configure_token.py
   ```

4. checkerを再実行する。credentialの実値はchat、コマンド引数、shell history、terminal output、log、commitへ残さない。

アカウント作成、保存先、client別の対応、troubleshootingは[認証設定の正本](references/kaggle-setup.md)を読む。
