# Reflect and Push

変更内容を適切なドキュメントファイルに反映してからプッシュします。

## Description
作業内容を関連するドキュメント（shared-status.md、implementation-plan.md等）に反映し、コミット・プッシュまでを一括実行します。

## Usage
```
/reflect-and-push [description]
```

## Examples
- `/reflect-and-push "魔力システム実装完了"`
- `/reflect-and-push "UI改良とバランス調整"`
- `/reflect-and-push "バグ修正とテスト追加"`

## What it does
1. 現在の作業内容を分析
2. `docs/shared-status.md` の進捗を更新
3. 必要に応じて `docs/implementation-plan.md` を更新
4. ブランチログファイルの作成/更新
5. 変更をコミット・プッシュ
6. TodoListがあれば適切に更新

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

1. 作業内容の分析と整理
2. 関連ドキュメントファイルの特定と更新
3. ブランチ別ログファイルの更新
4. コミットメッセージの自動生成
5. Git操作（add, commit, push）の実行
6. 進行状況の確認とレポート