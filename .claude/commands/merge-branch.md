# Merge Branch

他のブランチの変更をmainブランチに統合します。

## Description
指定されたブランチをmainブランチにマージし、必要に応じてブランチを削除します。project.godotのコンフリクトは自動的にmain側を優先して解決されます。

## Usage
```
/merge-branch [branch-name] [--delete-branch]
```

## Examples
- `/merge-branch feature/cow-personality` - ブランチをマージ（ブランチは保持）
- `/merge-branch feature/new-systems --delete-branch` - ブランチをマージ後に削除
- `/merge-branch bugfix/performance-issue --delete-branch` - バグ修正をマージ後に削除

## What it does
1. 指定されたブランチの存在確認
2. mainブランチに切り替え
3. リモートから最新情報を取得
4. 指定ブランチをmainにマージ
5. **project.godotのコンフリクト自動解決（main優先）**
6. マージコミットの作成
7. リモートへのプッシュ
8. オプション指定時にブランチ削除
9. `docs/shared-status.md`の更新

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

### 1. 事前チェック
- ブランチの存在確認
- 現在のブランチ状態確認
- 未コミット変更の確認

### 2. マージ処理
- `git checkout main` でmainブランチに切り替え
- `git fetch origin` でリモート情報を更新
- `git merge origin/[branch-name]` でマージ実行

### 3. コンフリクト自動解決
**project.godotの特別処理**:
```bash
# project.godotでコンフリクトが発生した場合
# main側の設定を優先して自動解決
git checkout --ours project.godot
git add project.godot
```

**その他のコンフリクト**:
- 手動解決が必要な場合は処理を停止
- 解決方法をユーザーに案内

### 4. 後処理
- マージコミットの作成
- `git push origin main` でリモート更新
- オプション指定時: `git branch -d [branch-name]` でローカルブランチ削除

## Options

### --delete-branch
マージ後にローカルブランチを削除します。
- リモートブランチは保持されます
- 削除前に確認メッセージを表示
- マージが成功した場合のみ実行

## 自動解決される項目

### project.godot
以下の項目でコンフリクトが発生した場合、main側を自動採用：
- `config/name` - ゲーム名
- `config/description` - ゲーム説明  
- `config/version` - バージョン
- その他のプロジェクト設定

### 理由
mainブランチは安定版として扱われるため、プロジェクト設定の統一性を保つためにmain側の設定を優先します。

## Error Handling

### よくあるエラーと対処法

1. **ブランチが存在しない**
   - ブランチ名を確認してください
   - `git branch -a` で利用可能なブランチを確認

2. **未コミット変更がある**
   - 現在の変更をコミットまたはstashしてください
   - `git status` で状態を確認

3. **複雑なコンフリクト**
   - project.godot以外でコンフリクトが発生した場合
   - 手動解決の手順を案内

## 更新されるファイル

マージ後に以下のファイルが自動更新されます：
- `docs/shared-status.md` - マージ状況の記録
- マージ対象ブランチのログファイル - 統合完了の記録

## Security Notes

- mainブランチの整合性を保つため、project.godotは自動的にmain側を採用
- 重要な変更は事前にレビューを推奨
- マージ前のバックアップは各自で実施

## Examples in Action

### 基本的なマージ
```
/merge-branch feature/magic-system
→ feature/magic-systemをmainにマージ、ブランチは保持
```

### マージ後にブランチ削除
```
/merge-branch bugfix/ui-fixes --delete-branch
→ バグ修正をマージ後、bugfix/ui-fixesブランチを削除
```

### コンフリクト発生時
```
project.godotでコンフリクト発生
→ 自動的にmain側の設定を採用
→ 「project.godotのコンフリクトを自動解決しました」と表示
```