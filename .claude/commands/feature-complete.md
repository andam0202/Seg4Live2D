# Feature Complete

機能実装の完了処理を行います。

## Description
機能実装が完了した際の一連の処理（テスト、ドキュメント更新、統合準備）を自動実行します。

## Usage
```
/feature-complete [feature-name]
```

## Examples
- `/feature-complete "魔力採集システム"`
- `/feature-complete "転生UI"`
- `/feature-complete "バグ修正#123"`

## What it does
1. テストとリントチェックを実行
2. 機能の完了をドキュメントに記録
3. TodoListの該当項目を完了にマーク
4. ブランチログに完了記録を追加
5. 統合準備状況をチェック
6. プルリクエスト準備（必要に応じて）

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

1. `/test-and-lint` の実行
2. 機能完了の確認とバリデーション
3. `docs/shared-status.md` での完了ステータス更新
4. `docs/branch-logs/[branch]/` に完了ログを追加
5. TodoListの完了マーク処理
6. 次の統合ステップの提案
7. 完了レポートの生成