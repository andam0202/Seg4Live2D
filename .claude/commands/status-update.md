# Status Update

プロジェクトの現在状況を更新・確認します。

## Description
現在の作業状況を分析し、関連するステータスファイルを更新します。また、全体の進捗状況を確認できます。

## Usage
```
/status-update [update-type]
```

## Examples
- `/status-update` (全体状況の確認)
- `/status-update progress` (進捗状況の更新)
- `/status-update todo` (TodoListの整理)
- `/status-update branch` (ブランチ状況の更新)

## What it does
1. 現在のGit状況を確認
2. TodoListの状況を確認・整理
3. `docs/shared-status.md` の情報を更新
4. ブランチ別ログファイルの状況確認
5. 全体的な進捗レポートを生成

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

1. Git状況の分析（`git status`, `git log`, `git branch`）
2. TodoListの読み込みと分析
3. 関連ドキュメントファイルの現在状況確認
4. 進捗状況の計算と更新
5. 次の作業項目の提案
6. 包括的なステータスレポートの生成