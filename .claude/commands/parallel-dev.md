# Parallel Dev

tmuxを利用して複数ブランチの並列開発環境を構築します。

## Description
複数のWorktreeで同時に開発を行うための並列開発環境をtmuxで構築します。上部にメインパネル、下部に2つの開発パネルを配置します。

### 推奨される使用フロー
1. **朝の開始時**: ユーザーが作りたいブランチをClaude Codeに要求
2. **Claude Codeが実行**: 
   - 必要なWorktreeを作成
   - parallel-devコマンドで3画面構成を構築
   - 各パネルで適切なディレクトリに移動
   - 下部パネルでClaude Codeを起動
3. **ユーザーが接続**: `tmux attach -t parallel-dev`で並列開発開始

## Usage
```
/parallel-dev [branch1-name] [branch2-name]
```

または

```
/parallel-dev
```
（ブランチ名を指定しない場合、現在の開発状況に基づいて適切なブランチ名を自動提案・作成）

## Examples
- `/parallel-dev magic/core-system magic/ui-integration`
- `/parallel-dev reincarnation/logic corruption/mechanics`
- `/parallel-dev` （自動提案）

## What it does
1. 指定された2つのブランチを作成（または自動生成）
2. それぞれのWorktreeパスを決定
3. tmuxセッションを作成
4. 画面を3分割（上部1画面、下部2画面）
5. 各パネルで対応するWorktreeに移動
6. 下部パネルでClaude Codeを起動準備

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

### ブランチ作成フェーズ
1. 2つのブランチ名の検証または自動生成
2. 各ブランチを作成（`git checkout -b [branch-name]`）
3. 各ブランチの専用ログディレクトリを作成
4. `docs/shared-status.md`に両ブランチ情報を追加

### Worktree作成フェーズ
ブランチタイプに応じて適切なWorktreeを作成：
```bash
# feature/*ブランチの場合
git worktree add ../nyugyu-clicker-feature feature/[branch-name]

# feature/*ブランチが複数ある場合の対応は別途検討

# bugfix/*ブランチの場合
git worktree add ../nyugyu-clicker-bugfix bugfix/[branch-name]

# experiment/*ブランチの場合
git worktree add ../nyugyu-clicker-experiment experiment/[branch-name]
```

**重要**: 各ブランチは独自のWorktreeディレクトリを持つ必要があります。同じWorktreeディレクトリに複数のブランチを配置することはできません。

### tmux環境構築フェーズ
```bash
# tmuxセッション作成
tmux new-session -d -s parallel-dev

# 上部パネル（メイン）
tmux send-keys -t parallel-dev:0.0 'cd /mnt/c/Users/mao0202/Desktop/Godot/projects/nyugyu-clicker' C-m
tmux send-keys -t parallel-dev:0.0 'echo "=== メインパネル (制御用) ==="' C-m

# 下部を2分割
tmux split-window -v -t parallel-dev:0
tmux split-window -h -t parallel-dev:0.1

# 画面レイアウト調整（上部30%、下部70%を左右均等分割）
tmux resize-pane -t parallel-dev:0.0 -y 30%

# 下部左パネル（ブランチ1） - 対応するWorktreeディレクトリに移動
tmux send-keys -t parallel-dev:0.1 'cd [WORKTREE_PATH_1]' C-m
tmux send-keys -t parallel-dev:0.1 'echo "=== [BRANCH_1] 開発環境 ($(pwd)) ==="' C-m
tmux send-keys -t parallel-dev:0.1 'claude' C-m

# 下部右パネル（ブランチ2） - 対応するWorktreeディレクトリに移動
tmux send-keys -t parallel-dev:0.2 'cd [WORKTREE_PATH_2]' C-m
tmux send-keys -t parallel-dev:0.2 'echo "=== [BRANCH_2] 開発環境 ($(pwd)) ==="' C-m
tmux send-keys -t parallel-dev:0.2 'claude' C-m

# tmuxセッションにアタッチ
tmux attach-session -t parallel-dev
```

## Worktreeパス決定ロジック
各ブランチタイプに応じて独自のWorktreeディレクトリを作成：

### ブランチタイプ別Worktree配置
- `feature/*` → `../nyugyu-clicker-feature`
- `bugfix/*` → `../nyugyu-clicker-bugfix`
- `experiment/*` → `../nyugyu-clicker-experiment`
- `test/*` → メインディレクトリ (`nyugyu-clicker`)
- `refactor/*` → メインディレクトリ (`nyugyu-clicker`)

### ブランチ命名規則の統一
**推奨命名規則**: すべてのブランチは `[カテゴリ]/[機能名]` の形式を使用
- UI改善: `feature/ui-main-game-improvements`
- バグ修正: `bugfix/save-load-issue`
- 実験的機能: `experiment/ai-driven-events`
- リファクタリング: `refactor/code-cleanup`

**非推奨**: `ui/main-game-improvements` のような独自カテゴリ
**理由**: Worktree管理が複雑化し、並列開発時に混乱を招く

## 自動ブランチ名生成
ブランチ名が指定されない場合、以下を分析して2つの関連ブランチを提案：
- 現在の実装フェーズ
- 未実装機能リスト
- 依存関係を考慮した並列実装可能な機能

例：
- `magic/core-system` と `magic/ui-integration`
- `reincarnation/logic` と `reincarnation/ui`
- `ui/status-panel` と `ui/responsive-design`

## 並列開発のベストプラクティス
1. **メインパネル活用**: 上部パネルでgit操作やファイル確認
2. **ブランチ間連携**: 共通ファイルの編集は調整が必要
3. **定期的な同期**: 定期的にメインブランチとの同期を推奨

## tmux基本操作ガイド
- **パネル間移動**: `Ctrl+b` → 矢印キー
- **パネルサイズ調整**: `Ctrl+b` → `Alt+矢印キー`
- **パネルを閉じる**: `exit` または `Ctrl+d`
- **セッションから離脱**: `Ctrl+b` → `d`
- **セッションに再接続**: `tmux attach -t parallel-dev`

## 注意事項
- tmuxがインストールされている必要があります
- 既存の`parallel-dev`セッションがある場合は警告を表示
- **重要**: 各ブランチには独自のWorktreeディレクトリが必要です
- 同じWorktreeディレクトリで複数のブランチを同時に使用することはできません
- Worktreeが存在しない場合は自動的に作成されます

## 日常的な使用例

### 朝の開発開始時の流れ
```
ユーザー: 「magic/core-systemとmagic/ui-integrationのブランチで並列開発したい」

Claude Code:
1. 必要に応じてWorktreeを作成
2. /parallel-dev magic/core-system magic/ui-integration を実行
3. 「tmuxセッションを作成しました。`tmux attach -t parallel-dev`で接続してください」

ユーザー:
- 別ターミナルでWSLを起動
- tmux attach -t parallel-dev を実行
- 3画面構成で並列開発を開始
```

### セッション管理
```bash
# 作業の中断
Ctrl+b → d  # セッションから離脱（作業は保持される）

# 作業の再開
tmux attach -t parallel-dev

# セッションの終了
tmux kill-session -t parallel-dev
```