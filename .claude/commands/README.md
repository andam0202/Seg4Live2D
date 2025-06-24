# Custom Commands Guide

**作成日**: 2025-06-22  
**対象プロジェクト**: nyugyu-clicker

このディレクトリには、開発効率を向上させるためのカスタムClaude Codeコマンドが含まれています。

## 利用可能なコマンド

### 🚀 ワークフロー系コマンド

#### `/commit-and-push` 
変更をコミット・プッシュ
```bash
/commit-and-push "feat: 新機能を追加"
```

#### `/reflect-and-push`
ドキュメント反映 → コミット・プッシュ
```bash
/reflect-and-push "魔力システム実装完了"
```

### 🌿 ブランチ管理系コマンド

#### `/setup-branch`
新しいブランチの作成とセットアップ
```bash
/setup-branch magic/core-system
```

#### `/sync-branch`
ブランチを最新状態に同期
```bash
/sync-branch
```

#### `/merge-branch`
他のブランチをmainに統合
```bash
/merge-branch feature/magic-system
/merge-branch bugfix/ui-fixes --delete-branch
```

### 📊 状況管理系コマンド

#### `/status-update`
プロジェクト状況の更新・確認
```bash
/status-update
/status-update progress
```

#### `/feature-complete`
機能実装完了処理
```bash
/feature-complete "魔力採集システム"
```

### 🔍 検索・品質系コマンド

#### `/quick-search`
プロジェクト内高速検索
```bash
/quick-search "CowData" code
/quick-search "魔力" docs
```

#### `/test-and-lint`
テスト・リントチェック実行
```bash
/test-and-lint
/test-and-lint gdscript
```

## コマンドの使い方

### 基本的な使用方法
Claude Codeセッション内で、`/[command-name]` の形式で実行します。

### よく使われるワークフロー

#### 1. 新機能開発の開始
```bash
/setup-branch magic/core-system
/status-update
```

#### 2. 実装中の状況確認
```bash
/quick-search "ManaSystem" code
/test-and-lint gdscript
```

#### 3. 機能完成時
```bash
/test-and-lint
/feature-complete "マナシステム基盤"
/reflect-and-push "魔力採集システム基盤完成"
```

#### 4. ブランチ同期
```bash
/sync-branch
/status-update branch
```

## コマンドの階層

### Level 1: 基本操作
- `/commit-and-push` - 単純なコミット・プッシュ
- `/status-update` - 状況確認

### Level 2: ワークフロー
- `/reflect-and-push` - ドキュメント更新込みプッシュ
- `/setup-branch` - ブランチセットアップ

### Level 3: 統合操作
- `/feature-complete` - 機能完了処理
- `/sync-branch` - ブランチ同期

## カスタマイズのヒント

### コマンドの追加
新しいコマンドを追加する場合：
1. `.claude/commands/[command-name].md` ファイルを作成
2. Description, Usage, Examples, What it does, Implementation セクションを記述
3. このREADMEに追加

### プロジェクト固有の調整
- Git操作はプロジェクトのブランチ戦略に合わせて調整
- ドキュメント更新は `docs/` 構造に合わせて実装
- テスト・リントはGodotプロジェクトの特性に合わせて設定

## 注意事項

### セキュリティ
- コマンドは現在のプロジェクト内でのみ動作
- 外部コマンドやシステム変更は含まない
- Git操作は安全な範囲内でのみ実行

### エラーハンドリング
- 各コマンドは実行前に状況を確認
- エラー時は適切なメッセージと回復手順を提示
- 破壊的操作は確認プロンプトを表示

### パフォーマンス
- 大きなプロジェクトでは検索範囲を適切に制限
- 並列実行可能な操作は最適化
- 重複処理を避けるための状態管理

---

このコマンドセットにより、従来の長い指示を短縮し、開発効率を大幅に向上できます。