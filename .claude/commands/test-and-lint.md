# Test and Lint

テストとリントチェックを実行します。

## Description
プロジェクトのテストスイートとコード品質チェック（リント）を実行し、問題があれば修正をガイドします。

## Usage
```
/test-and-lint [target]
```

## Examples
- `/test-and-lint` (全テスト・リント実行)
- `/test-and-lint gdscript` (GDScriptのみ)
- `/test-and-lint quick` (クイックチェックのみ)

## What it does
1. Godotプロジェクトの構文チェック
2. GDScriptの型チェック
3. ファイル構造の整合性確認
4. シーンファイルの依存関係チェック
5. エラーがあれば修正提案を生成

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

1. Godotプロジェクトファイルの検証
2. GDScriptファイルの構文・型チェック
3. シーンファイル（.tscn）の整合性確認
4. アセットファイルの参照チェック
5. エラー・警告の分析と修正提案
6. テスト結果のレポート生成