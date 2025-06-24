# Sync Branch

ブランチを最新状態に同期します。

## Description
現在のブランチを親ブランチ（main または feature/new-systems）の最新状態と同期し、競合を解決します。

## Usage
```
/sync-branch
```

## Examples
- `/sync-branch`

## What it does
1. 現在のブランチを確認
2. 親ブランチ（main or feature/new-systems）の最新変更を取得
3. マージまたはリベース実行
4. 競合があれば解決をガイド
5. `docs/shared-status.md` で同期状況を更新

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

1. 現在のブランチの親ブランチを特定
2. `git fetch origin` で最新情報を取得
3. 親ブランチの変更をマージ（`git merge origin/[parent-branch]`）
4. 競合が発生した場合は解決手順を表示
5. 同期完了後に状況をドキュメントに反映
6. 同期結果のレポート生成