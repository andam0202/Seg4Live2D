# Setup Branch

新しいブランチの作成とセットアップを行います。

## Description
階層的ブランチ管理に従って新しいブランチを作成し、必要な初期設定を行います。

## Usage
```
/setup-branch [branch-type]/[feature-name]
```

または

```
/setup-branch
```
（ブランチ名を指定しない場合、現在の開発状況に基づいて適切なブランチ名を自動提案・作成）

## Examples
- `/setup-branch magic/core-system`
- `/setup-branch reincarnation/ui`
- `/setup-branch bugfix/performance-issue`
- `/setup-branch ui/responsive-design`

## What it does
1. 指定されたブランチ名でブランチを作成（または自動生成）
2. ブランチ情報を `docs/shared-status.md` に追加
3. ブランチ専用ログディレクトリを作成
4. **会話内容を分析して作業目標を抽出**
5. **作業内容に基づいた初期TodoListを自動作成**
6. WSLターミナルを開いてプロジェクトディレクトリに移動
7. 作業開始の準備を完了
8. **新セッションでのコンテキスト自動継承を準備**

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

### ブランチ名が指定された場合
1. ブランチ名の検証（命名規則に従っているかチェック）
2. `git checkout -b [branch-name]` でブランチ作成

### ブランチ名が指定されない場合
1. 現在の開発状況を分析（プロジェクトファイル、docs/shared-status.md、実装計画を確認）
2. 以下の基準で適切なブランチ名を自動生成：
   - 現在のフェーズ・ステップに基づく機能名
   - 未完了のタスクや次に必要な実装内容
   - 既存ブランチとの重複を避ける
   - プロジェクトの命名規則に従う（[機能カテゴリ]/[具体的機能名]）
3. 提案されたブランチ名でブランチ作成

### 共通処理
3. `docs/branch-logs/[branch-name]/` ディレクトリ作成
4. `docs/shared-status.md` にブランチ情報を追加
5. **会話履歴分析**: 直近の会話から作業意図を抽出
6. **初期TodoList自動作成**: 抽出した作業内容に基づくタスク生成
7. ブランチタイプに応じた適切なWorktreeパスを決定
8. WSLターミナルを開いて対応するWorktreeディレクトリに移動
9. 作業開始レポートの生成

## 自動ブランチ名生成の基準

ブランチ名が指定されない場合の自動生成では、以下の分析を行います：

### 分析対象
- `docs/implementation-plan.md` - 現在のフェーズ・ステップ
- `docs/shared-status.md` - 既存ブランチと進捗状況
- `CLAUDE.md` - プロジェクト仕様と実装ステータス
- 現在のソースコード構造
- 最近のコミット履歴

### 生成例
- 現在フェーズが魔力システム実装中 → `magic/mana-collection`
- UI改善が必要 → `ui/status-layout-improvement`
- バグ修正が必要 → `bugfix/save-load-issue`
- テスト追加が必要 → `test/cow-system-validation`
- 転生システム未実装 → `reincarnation/core-mechanics`

### 命名規則
- `[カテゴリ]/[具体的機能名]`
- カテゴリ: `magic`, `reincarnation`, `ui`, `bugfix`, `test`, `performance`, `refactor`など
- 機能名: ケバブケース（小文字+ハイフン）

## VS Code Terminal セットアップ
VS Code Terminalの自動セットアップでは以下のコマンドを実行します：
```bash
# VS Codeの新しいウィンドウを指定されたディレクトリで開き、ターミナルを自動起動
cmd.exe /c "code --new-window [WORKTREE_WINDOWS_PATH] && timeout /t 1 && code --command workbench.action.terminal.toggleTerminal"
```

Worktreeパス対応（Windows → WSL）：
- `main`:
  - Windows: `C:\Users\mao0202\Desktop\Godot\projects\nyugyu-clicker`
  - WSL: `/mnt/c/Users/mao0202/Desktop/Godot/projects/nyugyu-clicker`
- `feature/*`:
  - Windows: `C:\Users\mao0202\Desktop\Godot\projects\nyugyu-clicker-feature`
  - WSL: `/mnt/c/Users/mao0202/Desktop/Godot/projects/nyugyu-clicker-feature`
- `bugfix/*`:
  - Windows: `C:\Users\mao0202\Desktop\Godot\projects\nyugyu-clicker-bugfix`
  - WSL: `/mnt/c/Users/mao0202/Desktop/Godot/projects/nyugyu-clicker-bugfix`
- `experiment/*`:
  - Windows: `C:\Users\mao0202\Desktop\Godot\projects\nyugyu-clicker-experiment`
  - WSL: `/mnt/c/Users/mao0202/Desktop/Godot/projects/nyugyu-clicker-experiment`
- `test/*`:
  - Windows: `C:\Users\mao0202\Desktop\Godot\projects\nyugyu-clicker` (メイン)
  - WSL: `/mnt/c/Users/mao0202/Desktop/Godot/projects/nyugyu-clicker`
- `refactor/*`:
  - Windows: `C:\Users\mao0202\Desktop\Godot\projects\nyugyu-clicker` (メイン)
  - WSL: `/mnt/c/Users/mao0202/Desktop/Godot/projects/nyugyu-clicker`

このコマンドにより：
1. VS Codeの新しいウィンドウが対応するWorktreeディレクトリで開く
2. 1秒待機後、自動的にVS Codeのターミナルが開く
3. そのVS Codeターミナルで手動で `wsl` コマンドを実行
4. WSLが起動され、正しいプロジェクトディレクトリに配置される
5. そこで手動で `claude` コマンドを実行してClaude Codeを起動
6. VS Codeターミナルではクリップボードのペーストが正常に動作

## 会話内容分析と自動Todo作成

### 分析対象
setup-branchコマンド実行前の会話履歴から以下を抽出：
- 作業目標・実装したい機能
- 修正・改良したい内容
- 特定のタスクや要求

### 自動Todo生成パターン
抽出した内容に基づいて以下の形式でTodoを生成：
```
- [ ] [抽出した機能名]の設計・仕様確認
- [ ] [抽出した機能名]の実装
- [ ] テスト・動作確認
- [ ] ドキュメント更新
```

### 使用例
**会話例**:
```
ユーザー: 「魔力採集システムの自動化機能を実装したい。/setup-branch」
```

**生成されるTodo**:
```
- [ ] 魔力採集システムの自動化機能の設計・仕様確認
- [ ] 魔力採集システムの自動化機能の実装
- [ ] テスト・動作確認
- [ ] ドキュメント更新
```

## 新セッションでのコンテキスト継承

新しいClaude Codeセッションが開始されると：
1. 自動的に`TodoRead`を実行
2. 作成されたTodoリストを表示
3. 「以下のタスクに取り組みます」と開始メッセージを表示
4. ユーザーは追加説明不要で即座に作業開始可能

## 手動ステップ
VS Codeの新しいウィンドウが開き、ターミナルが自動起動したら、以下を実行：

### 1. VS CodeターミナルでWSLを起動（ターミナルは自動で開きます）
```powershell
wsl
```

### 2. WSLでClaude Codeを起動
```bash
claude
```

**自動継承により、新セッションではコンテキストが既に設定済み**
**VS Codeターミナルではクリップボードペーストが正常動作**

### 実行フロー
1. `/setup-branch` コマンド実行
2. VS Codeの新しいウィンドウが指定されたWorktreeディレクトリで起動
3. 1秒待機後、VS Codeターミナルが自動で開く
4. VS Codeターミナルで手動で `wsl` コマンドを実行
5. WSLが起動され、正しいプロジェクトディレクトリに配置
6. WSLプロンプトで手動で `claude` コマンドを実行
7. 新しいClaude Codeセッションが自動Todoと共に開始（クリップボード対応）