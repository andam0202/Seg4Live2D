# Commit and Push

変更をコミットしてプッシュします。

## Description
現在のワーキングディレクトリの変更を自動的にステージング、コミット、プッシュする一連の作業を実行します。

## Usage
```
/commit-and-push [commit message]
```

## Examples
- `/commit-and-push "feat: 新機能を追加"`
- `/commit-and-push "fix: バグを修正"`
- `/commit-and-push "docs: ドキュメントを更新"`

## What it does
1. `git status` で現在の状態を確認
2. `git add .` で全ての変更をステージング
3. `git commit` でコミット（メッセージが提供されない場合は自動生成）
4. `git push origin [current-branch]` でプッシュ
5. 結果を確認

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

1. Git状態確認
2. 変更されたファイルの自動ステージング
3. 適切なコミットメッセージの生成（必要に応じて）
4. プッシュ実行
5. 結果の確認とレポート