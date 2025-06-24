# Quick Search

プロジェクト内での高速検索を行います。

## Description
プロジェクト内のファイル、コード、ドキュメントを効率的に検索し、関連情報を素早く取得します。

## Usage
```
/quick-search [search-term] [scope]
```

## Examples
- `/quick-search "CowData" code` (コード内検索)
- `/quick-search "魔力" docs` (ドキュメント内検索)
- `/quick-search "*.gd" files` (ファイル名検索)
- `/quick-search "signal" all` (全体検索)

## What it does
1. 指定されたスコープで検索を実行
2. 検索結果を整理して表示
3. 関連ファイルの一覧生成
4. コードの場合は関数・クラス情報も表示
5. 検索結果から関連項目を提案

## Implementation
このコマンドを実行すると、以下の処理を自動的に行います：

1. 検索スコープの決定（code, docs, files, all）
2. 適切な検索ツール（Grep, Glob）の選択と実行
3. 検索結果のフィルタリングと整理
4. ファイル内容の要約（該当部分の抜粋）
5. 関連ファイル・機能の提案
6. 検索結果レポートの生成