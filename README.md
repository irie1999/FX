# FX アルゴリズムトレード / バックテスト

シンプルな FX 取引アルゴリズム（SMA クロス + RSI フィルタ）と、ベクトル化バックテスト環境。

## 構成

```
src/fx/
  data.py       # CSV / 合成データのロード
  strategy.py   # シグナル生成 (SMA cross + RSI)
  backtest.py   # ベクトル化バックテスト (スプレッド考慮)
  metrics.py    # 損益・シャープ・最大DD・勝率
  main.py       # CLI エントリーポイント
tests/          # pytest テスト
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

合成データでサンプル実行:

```bash
python -m fx.main --synthetic --bars 5000
```

CSV を使う場合 (列: `timestamp,open,high,low,close`):

```bash
python -m fx.main --csv path/to/usdjpy_1h.csv
```

HistData.com の Generic ASCII M1 をそのまま読み込む場合 (ファイル移動不要):

```bash
# 単一ファイル
python -m fx.main --histdata "C:/Users/you/Downloads/DAT_ASCII_USDJPY_M1_2024.csv"

# 複数ファイル / glob / ディレクトリ指定もOK
python -m fx.main --histdata "C:/Users/you/Downloads/DAT_ASCII_USDJPY_M1_*.csv" --resample 1h

# 1 分足を 5 分足 / 1 時間足 / 日足にリサンプル
python -m fx.main --histdata "C:/Users/you/Downloads/" --resample 1h
```

HistData のタイムスタンプは EST（DST なし、UTC-5）なので内部で UTC に変換されます。

### HistData を自動ダウンロード

ブラウザで手動 DL しなくても、`tools/fetch_histdata.py` がまとめて取得します（stdlib のみ、追加 pip install 不要）。

```bash
# 完了済みの年を年次 ZIP で取得
python tools/fetch_histdata.py --pair USDJPY --years 2023 2024

# 今年分は月単位で取得
python tools/fetch_histdata.py --pair USDJPY --months 2026-01 2026-02 2026-03

# そのままバックテストへ
python -m fx.main --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" --resample 1h
```

`--out` で保存先を変えられます。既に同名 CSV があればスキップします。

## グリッドサーチ

`tools/grid_search.py` でパラメータ空間を総当たりし、PF / Sharpe / MAR などでランキング。

```bash
python tools/grid_search.py --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --resample 1d --equity 500000 \
    --fast 10,15,20,30 --slow 30,50,75,100 \
    --rsi-period 10,14,21 \
    --stop-atr none,1.0,1.25,1.5,2.0 \
    --adx-threshold 0,20,25,30 \
    --top 30 --sort-by pf \
    --html results/grid.html --csv-out results/grid.csv
```

- カンマ区切りで各軸の候補値を指定
- `--stop-atr` に `none` を含めると「ストップ無効」も比較対象
- `--sort-by` は `pf` / `sharpe` / `cagr` / `mar` / `net_profit` から選択
- `--html` で上位 N 件をダークテーマ HTML で表示（`--no-open` でブラウザ起動抑止）
- `--csv-out` で全組み合わせの結果を CSV 保存（分析・可視化に利用）
- `fast >= slow` や `rsi_lower >= rsi_upper` の組み合わせは自動除外

## ウォークフォワード検証

`tools/walk_forward.py` は「各窓で再最適化 → 次の窓で OOS 検証」を繰り返し、**カーブフィットを見抜く**ための仕組みです。

```bash
python tools/walk_forward.py --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --resample 1d --equity 500000 \
    --train 540 --test 180 --step 180 \
    --fast 10,15,20,30 --slow 50,75,100 \
    --rsi-period 14,21 --rsi-upper 60,70 --rsi-lower 30 \
    --stop-atr 1.0,1.25,1.5,2.0 \
    --sort-by pf --html results/wfa.html
```

- `--train` / `--test` / `--step` は **バー数**（`--resample` 後の本数）
- 各窓: train 範囲でグリッドサーチ → ベストパラメータを test で OOS 検証
- 全 test 区間を繋いで **OOS 資産推移** と集計指標を算出
- HTML レポートには窓ごとの詳細 + パラメータ安定性（どのパラメータが何回選ばれたか）も含む
- 判定: **OOS PF > 1.5 なら本物**、OOS PF < 1.0 ならカーブフィット

## 直近データの取得 (yfinance)

HistData は **完了月** までしか配信しないので、現在月のデータは取得できません。Yahoo Finance なら**昨日まで**の日足が取れるので、それを HistData に上乗せして使います。

```bash
pip install yfinance   # 初回のみ

# 1 通貨だけ取得
python tools/fetch_recent.py --pair USDJPY --period 60d

# 5 通貨まとめて取得（PowerShell）
foreach ($pair in 'USDJPY','EURUSD','GBPUSD','AUDUSD','EURJPY') {
    python tools/fetch_recent.py --pair $pair --period 90d
}
```

`data/recent/<PAIR>.csv` に保存されます。これを HistData に上乗せして使う:

```bash
# 単一通貨のシグナル（直近データ込み）
python tools/daily_signal.py \
    --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --extra-csv "data/recent/USDJPY.csv" \
    --resample 1d --strategy adaptive --size 1000 --stop-atr 1.25

# マルチ通貨バックテスト（直近データ込み）
python tools/multi_pair_backtest.py \
    --histdata-dir data/raw \
    --extra-dir data/recent \
    --pairs USDJPY,EURUSD,GBPUSD,AUDUSD,EURJPY \
    --resample 1d --strategy adaptive --stop-atr 1.25 \
    --size 1000 --equity-per-pair 100000 --eval-days 30 \
    --html results/multi_pair_30d.html
```

`merge_recent` は同じタイムスタンプで重複した場合 **recent 側を優先** するので、HistData の境界バーが yfinance の値で上書きされる挙動になります。

## 学術ベースの追加戦略

論文ベースで実装した、より高い利益を狙える戦略 3 つ。

### 1. 時系列モメンタム (TSMOM)

Moskowitz, Ooi, Pedersen (2012) *"Time Series Momentum"* (JFE)

過去 N バーの累積リターンの正負で次バーの方向を予測。学術的に最強クラスの単純戦略 (Sharpe 1.5、年率 17% を 1985-2009 で実証)。

```bash
python -m fx.main --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --resample 1d --strategy tsmom --stop-atr 1.25 --size 1000 --html
```

### 2. BB スクイーズ・ブレイクアウト (BBS)

ボリンジャーバンドの幅が直近最低水準（スクイーズ）まで縮んだ後の急騰急落を狙う。

```bash
python -m fx.main --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --resample 1d --strategy bbs --stop-atr 1.25 --size 1000 --html
```

### 3. 統計的裁定取引 (Pairs Trading)

Gatev et al. (2006), Avellaneda & Lee (2010)。相関の高い 2 通貨ペアの「ズレ」を取る market-neutral 戦略。

```bash
python tools/pairs_trading.py \
    --histdata-dir data/raw \
    --pair-a EURUSD --pair-b GBPUSD \
    --resample 1d --size 1000 --equity 200000 \
    --z-window 60 --entry-z 2.0 --exit-z 0.5 \
    --html results/pairs_eurgbp.html
```

z スコアが ±2.0 を超えたらスプレッドを売買、|z| < 0.5 で利確、|z| > 4.0 で逆指値。市場全体の方向性に関係なく動くので、他戦略と相関ゼロの分散効果。

### 比較

```bash
python tools/compare_strategies.py --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --resample 1d --equity 500000 --stop-atr 1.25 --swing \
    --strategies sma_rsi,tsmom,bbs,supertrend,ichimoku,adaptive \
    --html results/swing_compare_extended.html
```

## マルチ通貨ポートフォリオ

複数の通貨ペアで同じ戦略を並列に走らせ、ポートフォリオとしての成績を計測します。**取引機会と分散効果**の両方を得られます。

```bash
# 過去データを各通貨で取得
python tools/fetch_histdata.py --pair USDJPY --years 2020 2021 2022 2023 2024 2025
python tools/fetch_histdata.py --pair EURUSD --years 2020 2021 2022 2023 2024 2025
python tools/fetch_histdata.py --pair GBPUSD --years 2020 2021 2022 2023 2024 2025
python tools/fetch_histdata.py --pair AUDUSD --years 2020 2021 2022 2023 2024 2025
python tools/fetch_histdata.py --pair EURJPY --years 2020 2021 2022 2023 2024 2025

# マルチ通貨バックテスト（ポートフォリオ）
python tools/multi_pair_backtest.py \
    --histdata-dir data/raw \
    --pairs USDJPY,EURUSD,GBPUSD,AUDUSD,EURJPY \
    --resample 1d --strategy adaptive --stop-atr 1.25 \
    --size 1000 --equity-per-pair 100000 \
    --html results/multi_pair.html --csv-out results/multi_pair.csv
```

仕様:
- 各通貨ペアに `--equity-per-pair` (default ¥100,000) を独立配分
- 各ペアの非 JPY PnL は概算レート（USD≈¥150, EUR≈¥165 等）で JPY 換算
- ポートフォリオとして合計エクイティ・Sharpe・MaxDD を算出
- HTML レポートは通貨別+合計の資産推移チャート、通貨別パフォーマンス表

スプレッドは SBI 風のデフォルト値を使用、上書きしたい場合:
```
--spread-overrides "EURUSD=0.0001,GBPUSD=0.0002"
```

## 毎朝のシグナル出力（手動発注向け）

SBI FX トレード等の自動売買 API が無い業者で運用する場合、**毎朝アルゴリズムの指示を確認 → 手でアプリから発注** というフローが現実的です。そのための専用ツール:

```bash
python tools/daily_signal.py \
    --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --resample 1d --strategy adaptive \
    --size 1000 --stop-atr 1.25 --spread 0.02 \
    --output results/signal.txt \
    --json results/signal.json
```

出力は「現在の状態」と「翌バーで何をするか」を SBI アプリの操作手順付きで表示:

```
============================================================
 📅 FX 取引シグナル — 作成: 2026-04-02 07:00
============================================================
 通貨ペア      : USD/JPY
 データ最終    : 2026-04-01 00:00:00+00:00
 現在値        : 158.8500
 現在のポジション: ロング 1,000 通貨
   エントリー日  : 2025-10-17
   エントリー値  : 150.5830
   含み損益      : ¥+8,267
   ストップ水準  : 148.7500

------------------------------------------------------------
 📌 翌バーのアクション: ロング継続
------------------------------------------------------------
  → 既存ポジションを継続保有。ストップ注文はそのまま維持。
============================================================
```

新規エントリー時は SBI アプリでの具体的な発注手順が出ます:
```
 【SBI FX アプリでの手順】
  1. USD/JPY を選択
  2. 『新規』成行注文
  3. 『買い』 1,000 通貨
  4. 逆指値 (ストップ) を 156.500 にセット

 💰 想定リスク (ストップ到達時): ¥-2,350
```

### Discord / Slack へ自動通知

LINE Notify は 2025-03 で終了したので後継として Discord または Slack を使います。どちらも Webhook URL を 1 行貼り付けるだけ:

```bash
python tools/daily_signal.py \
    --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --resample 1d --strategy adaptive --size 1000 --stop-atr 1.25 \
    --webhook "https://discord.com/api/webhooks/xxxxxx/yyyyyy"
```

### 夜間実行のタイミング（推奨: 22:30 JST）

FX 日足は NY クローズ (21:00〜22:00 UTC = JST 翌朝 06:00〜07:00) で締まります。**夜 22:30 JST に実行する場合**のタイムライン:

```
  ← 過去の日足バー (最終)
  |=======================|-----今日はここ----|=======================|
  06:00 JST (今朝)        22:30 JST (実行!)    06:00 JST (明朝, 次バー締め)
                          あと 7.5h で次バー締める
```

- 実行時点の**最新確定バー = 今朝 06:00 JST に締まった分**
- シグナルは「**次の日足バー** (今夜〜明朝の値動き) でのポジション希望」を表す
- NY 市場が開いた直後で流動性が高い → **このまま SBI アプリで発注**して就寝、明朝の値動きは自動で反映
- ストップ注文は GTC (Good Till Cancelled) で置いておけば、寝ている間に逆行しても自動決済

### Windows Task Scheduler で夜間自動実行

1. データ更新: 月初に `python tools/fetch_histdata.py --pair USDJPY --months YYYY-MM` を走らせるタスク
2. シグナル生成: 平日 22:30 (JST) に上記コマンドを走らせるタスク
3. Webhook 通知で携帯に届く → そのまま SBI アプリで発注して就寝

タスク作成例 (PowerShell、夜 22:30 JST 実行):
```powershell
$action = New-ScheduledTaskAction -Execute "python" `
    -Argument "tools\daily_signal.py --histdata data\raw\DAT_ASCII_USDJPY_M1_*.csv --resample 1d --strategy adaptive --size 1000 --stop-atr 1.25 --webhook YOUR_URL" `
    -WorkingDirectory (Get-Location)
$trigger = New-ScheduledTaskTrigger -Daily -At 10:30pm
Register-ScheduledTask -TaskName "FX Nightly Signal" -Action $action -Trigger $trigger
```

朝に実行したい場合は `-At 7:05am` に変更するだけ。出力自体は実行時刻に依存しません。

## デイトレ (ライブ / ペーパー)

`tools/live_trade.py` は戦略を「今の相場」で回し、OANDA 経由で発注します。デフォルトは**ペーパー（模擬）**、実運用は明示的に `--broker oanda --live` を付けた場合のみ。

```bash
# 1) ペーパートレード（データ取得不要で挙動確認）
python tools/live_trade.py --broker paper --once

# 2) OANDA Japan のデモ口座で実発注（無料、無リスク）
export OANDA_TOKEN=xxxxxxxx
export OANDA_ACCOUNT=101-001-xxxxxxx-001
python tools/live_trade.py --broker oanda --instrument USD_JPY --granularity M15 --size 10000

# 3) 本番口座で実取引（自己責任！）
python tools/live_trade.py --broker oanda --live --instrument USD_JPY --granularity M15 --size 10000
```

### OANDA Japan デモ口座のセットアップ

1. https://www.oanda.jp/ でデモ口座を作成（無料、本人確認不要）
2. 口座管理画面で **API token** を発行
3. 口座 ID（`101-001-xxxxxxx-001` 形式）を控える
4. 環境変数 `OANDA_TOKEN` と `OANDA_ACCOUNT` にセット
5. 上記コマンド実行 → ログは `logs/live.log` に記録

### デイトレ特有のルール

- **EOD 強制決済**: デフォルト 21:00 UTC（NY クローズ）で全ポジション flat
- オプション: `--eod-utc 23:00` で時刻変更
- 15 分足デフォルト (`--granularity M15`); M1/M5/M30/H1/H4 も選択可
- 1 バー確定ごとに自動ループ（`--once` で 1 回のみ）

### 重要な安全装置

- `--broker oanda` 指定時、デフォルトは `OANDA_ENV=practice`（デモ口座）
- **実資金発注には必ず `--live` が必要**
- `--dry-run`: シグナルだけ計算して発注しない
- `--once`: 1 イテレーションで終了（cron / 手動検証用）
- 全発注・シグナル・例外を `logs/live.log` に記録

### ⚠️ 免責と注意

- **このツールは教育・研究目的で提供**されています
- FX 実取引には**スリッページ、約定拒否、サーバー障害、スプレッド拡大**等のリスクあり
- 戦略のバックテスト成績は**将来の収益を保証しません**
- デモで最低 **1〜3 ヶ月**動かし、バックテストと成績が一致することを確認してから少額の実資金を検討してください
- 本ツールの利用による損失について作者は一切責任を負いません

## HTML レポート出力

バックテスト結果を自己完結型の HTML レポートとして書き出せます（チャートは base64 PNG で埋め込み、外部リソース依存なし）。

```bash
# デフォルトの保存先 results/report.html、終わったら既定ブラウザで自動オープン
python -m fx.main --synthetic --bars 5000 --html

# 保存先を指定
python -m fx.main --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" --resample 1h \
    --html results/usdjpy_1h.html --title "USDJPY 1h backtest"

# ブラウザを開きたくない場合
python -m fx.main --synthetic --html --no-open
```

レポートに含まれる内容:
- パフォーマンス指標（Total Return / Sharpe / Max DD / 勝率 / PF ほか）
- パラメータ・期間・スプレッド設定
- エクイティカーブとドローダウン
- 価格チャート（SMA ・売買シグナルのマーカー付き）
- トレード別 PnL の棒グラフ
- トレード一覧（先頭 50 件）

主要オプション:

- `--fast` 短期 SMA 期間 (default 20)
- `--slow` 長期 SMA 期間 (default 50)
- `--rsi` RSI 期間 (default 14)
- `--rsi-upper` / `--rsi-lower` RSI フィルタの閾値
- `--stop-atr N` ATR ベースのストップロス (N × ATR 距離で自動決済)
- `--adx-threshold N` ADX > N の時だけエントリー (トレンド強度フィルタ、0 = 無効)
- `--adx-period N` ADX 算出期間 (default 14)
- `--currency SYM` HTML に表示する通貨記号 (default ¥、EURUSD 等は `$` 推奨)

バックテスト実行後、コンソール末尾と HTML 冒頭に **現在の状態 / 翌バーのアクション** が表示されます。
- 現在のポジション（ロング / ショート / フラット）
- 含み損益・エントリー日時・ストップ水準
- 翌バーのシグナルと具体的な発注指示 (「新規ロング」「ショートへドテン」等)
- `--spread` スプレッド (price units)
- `--size` 1 トレードの枚数
- `--start YYYY-MM-DD` / `--end YYYY-MM-DD` バックテスト期間の絞り込み
  例: `--start 2024-01-01 --end 2024-12-31` で 2024 年のみ検証

## テスト

```bash
PYTHONPATH=src pytest -q
```
