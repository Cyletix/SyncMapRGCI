# SyncMapRGCI

[English](README.md) · [简体中文](README.zh-CN.md) · **日本語**

SyncMapRGCI は、自己組織化による因果推論を目指す研究プロジェクトです。目指しているのは、ラベルなしで系列に隠れた階層的なチャンク構造を発見し、さらにそれらのチャンク間の因果関係を推論するシステムです。長期的な対象は視覚的なイベント系列です（研究テーマ：*Visual Causal Inference through Hierarchical Chunking via Self-Organizing Dynamics*）。本プロジェクトは、Vargas研究室（Laboratory of Intelligent Systems, LIS）の Elrion プロジェクトにおける SyncMap と SyncMapX を基盤としています。これは私個人の研究・学習用の保管庫で、2025年10月に始めました。

プロジェクトは次の 2 つの部分から成ります：

- **RG, Recursive Grouping（再帰的グルーピング、実装済み）。** 1 つの幾何空間の中での階層的チャンキングです。同じ自己組織化カーネルを、下位レベルが生成したトークンに対して、より大きな自然スケール（レベルごとに x3.16）で再び適用します。これにより、チャンクとチャンクのグループが、1 つのマップの中の入れ子になったクラスタとして現れます。
- **CI, Causal Inference（因果推論、初期実験段階）。** 有向関係を扱います：有向遷移統計、合成 DAG 上のエッジ復元、ライングラフ / エッジトークン表現。

両者の統合、すなわち RG が発見したチャンク間の因果関係は、粗いプロトタイプとしてしか存在せず、このリポジトリには含まれていません。現在の実験はすべて、合成グラフウォークによるシンボル系列を用いています。`.dot` グラフ上のランダムウォークが入力ストリームを生成し、各モデルはグラフのチャンク（フラット）または入れ子になったチャンクのレベル（階層）を復元しなければなりません。階層グラフでは、ウォークはノードの多レベルラベルから生成されます。ウォークはほとんどの場合チャンク内にとどまり、ときどき兄弟チャンクへ移り、まれに別のグループへ移ります。ラベルはスコア計算にのみ使用されます。

![DCH1 の隠れた階層と、RG がウォークのみから学習したマップ](results/benchmark/figures/hero_rg_DCH1.png)

*左：ウォークを生成する隠れた構造。右：100,000 ステップ後の RG のマップ（RG 自身の 3 次元座標）。2 つのグループは大きく離れており、各グループを拡大すると、そのチャンクがより小さなクラスタとして見えます。*

**初めての方へ：** [プレゼンテーション](docs/presentation/index.html)（スピーカーノート付きの HTML スライド。単一ファイル版と PDF は [リリースページ](https://github.com/Cyletix/SyncMapRGCI/releases) にあります）では、元の SyncMap から結果まで、RG を順を追って説明しています。その他の資料はすべて [ドキュメントマップ](docs/README.md) に一覧があります。

## アルゴリズム

主な研究ライン：

- `NSGD`（`nsgd`）：一次の SyncMap ダイナミクス。指数重み付きの共活動統計が、しきい値付き k 近傍の関係グラフを定めます。位置は関係による引力と活動依存の変位によって動き、点ごとの可塑性と上限付きのステップを持ちます。ベンチマークで最も強いベースラインです。ファイル：`syncmap/SyncMapX_NSGD.py`。
- `NS`（ランタイム名 `ns`、`--strategy ns` で選択。プリセットなし）：NSGD から発展した、階層に対応した一次カーネル。単体でも動作し、RG の各レベルのエンジンでもあります。その挙動はもはや NSGD と同一ではありません。ファイル：`syncmap/SyncMapNS.py`。
- `RG`（`rg`）：`SyncMapNS` 上に構築された再帰的な階層拡張。レベル 0 は生のシンボルを学習し、上位の各レベルは下位レベルを教師なしで粗視化して得られたトークンを受け取ります。距離、更新間隔、統計の記憶長はレベルが上がるごとに幾何級数的に増加します。主な力は、プリミティブ対照力、遷移グラフのばね、局所的な障害物反発、上限付きのマクロフィードバックです。RG は独自のレベルごとの分割も保持しており（`cluster_level1` は元のシンボル上のレベル 0、`cluster_levelN` はレベル N-1 を元のシンボルへ引き戻したもの）、ビューアとスナップショット JSON に表示されますが、スコア計算には使われません。ファイル：`syncmap/SyncMapRG.py`。
- `CI`（ランタイム名 `ci`、`--strategy ci` で選択。プリセットなし）：RG とは独立に動作する有向遷移ベースライン。`syncmap/SyncMapCI.py` は有向遷移統計、エッジ復元の評価、非対称なノード特徴、ライングラフ / エッジトークン表現を提供し、`syncmap/SyncMapX_CI.py` はそれらの上に構築したプロトタイプ埋め込みをクラスタリングします。これは研究用の足場であり、完成した因果推論手法ではありません。

比較用ベースライン：

- `SyncMap`（`syncmap`）：一定の学習率と独自の DBSCAN 読み出しを持つ、オリジナルの SyncMap。ファイル：`syncmap/SyncMap.py`。
- `Symmetric`（`symmetric`）：記憶ウィンドウ、対称な正例/負例サンプリング、独自の移動平均読み出しを持つ Symmetrical SyncMap。ファイル：`syncmap/SyncMapX_Symmetric.py`。
- `TSF`（`tsf`）：TSFMap 方式の重心ダイナミクスで、独自の事後的な凝集型階層読み出しを持ちます。ファイル：`syncmap/SyncMapTSF.py`。

各モデル独自の読み出しはビューアで使われます。このリポジトリのスコアはすべて、座標に対する共通の 1 つの読み出しから得られています（[評価](#評価) を参照）。

## 由来

この研究は、九州大学の Danilo Vasconcellos Vargas 教授の研究室で、研究室内部の Elrion フレームワーク（未公開）の上に構築する形で始まりました。このリポジトリには私自身のコードと合成データセットのみが含まれており、未公開の Elrion のコンポーネントは含まれていません。

比較用ベースラインは公開済みの研究に基づいています：

- SyncMap：Vargas & Asabuki, "Continual General Chunking Problem and SyncMap", AAAI 2021。`syncmap/SyncMap.py` は https://github.com/zweifel/SyncMap のリファレンス実装（MIT License, Copyright (c) 2021 zweifel）を移植したもので、`test/` 以下のグラフウォーク環境とチャンク環境も同じリリースに由来します。
- Symmetrical SyncMap：Zhang & Vargas, "Symmetrical SyncMap for Imbalanced General Chunking Problems", arXiv:2310.10045。
- TSFMap：Vargas, Tham & Zhang, "Dynamical Equations with Bottom-up Self-Organizing Properties Learn Accurate Dynamical Hierarchies without any Loss Function", arXiv:2302.02140, 2023。`syncmap/SyncMapTSF.py` はそのダイナミクスを独自に再実装したものです。

## インストール

Python 3.11 を使用します。`pygraphviz`（`.dot` データセットの読み込みに使用）には、事前に Graphviz の C ライブラリとヘッダーが必要です。

```bash
sudo apt install graphviz graphviz-dev
conda create -n syncmaprgci python=3.11
conda activate syncmaprgci
pip install -r requirements.txt
```

root 権限がない場合は、`conda install -c conda-forge pygraphviz` で Graphviz を `pygraphviz` と一緒にインストールできます。`PyQt5` とディスプレイが必要なのは対話型ビューア（タスク 1、および `--live` 付きのタスク 0）だけです。ヘッドレス環境では `MPLBACKEND=Agg` を設定してください。

## 使い方

`main.py` はすべての実験タスクを実行します。`tools/` 内のスクリプトは、その上にベンチマーク、補足実験、およびそれらのレポートを構築します。リポジトリのルートから実行してください。データセットのパスでは `datasets/` を省略できます。

5 つのモデルすべてが深い階層を学習する様子を、リアルタイムの ARI/NMI 曲線とともに並べて観察します（Qt とディスプレイが必要）：

```bash
python main.py --task 1 --strategy syncmap symmetric tsf nsgd rg --dataset hierarchy/IEH_fixed_chunk_deep_hierarchy.dot --steps 1000000 --record-interval 5000 --seed 0
```

DCH1 上で RG を 1 回実行し、最終マップ（PNG）、学習の軌跡（GIF）、ARI 曲線を `results/picture/`、`results/animations/`、`results/snapshots/` に書き出します。出力される `result:` 行は `... ari=0.8836 ... L1=0.7671 L2=1.0000` となります：

```bash
python main.py --task 0 --strategy rg --dataset hierarchy/DCH1_fixed_chunk_6_3.dot --steps 100000 --seed 0 --save-png true --save-gif true --trace true --save-plot true
```

`datasets/` 内の 39 個のフラットグラフと、10 グラフからなる階層スイート（`--datasets flat hierarchy`、計 49 グラフ）に対する一括比較を 5 つのシードで行い、CSV と Markdown のレポートを `results/experiment/<timestamp>/` に出力します（長時間かかります）：

```bash
python main.py --task 2 --datasets flat hierarchy --profiles syncmap symmetric tsf nsgd rg --comparison-sequence-lengths 5000 100000 --comparison-trials 5 --metric ari --seed 0
```

以下に示すベンチマークを再構築します（8 グラフ、5 モデル、5 シード、図と GIF を含む）：

```bash
python tools/benchmark.py run
```

`--set key=value` でその実行に限りモデルのパラメータを 1 つ上書きでき、`python main.py --list-profiles` でプリセットの一覧を表示できます。すべてのタスクとフラグは [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) にまとめてあります。

## 実験パラメータ

共通の入力環境：

- マップの次元：3。
- 入力エンコーダ：各ノードは 11 入力ステップの間保持されます（`TIME_DELAY = 10`）。入力ベクトルは、減衰 `exp(-0.1 t)` を伴う現在のノードに加えて、直前のノードのより弱い痕跡を含みます。階層グラフでは、ウォークは確率 0.45 でチャンク間を移動し、k レベル上のグループ間は確率 0.45 × 0.25^k で移動します（[詳細](docs/DESIGN_REFERENCE.md#input-environments)）。
- 学習率（`adaptation_rate`）：NSGD と RG は 0.05、TSF は 0.001。SyncMap と Symmetric は `clip(0.001 x number_of_symbols, 0.05, 0.1)` を使います。`--set adaptation_rate=...` でいずれも上書きできます。

プロファイルのパラメータ（`tools/main_config.py`、`VARIANT_CONFIGS`）。ここに挙げる読み出しの設定は各モデル独自の読み出しのものであり、スコア計算用の読み出しではありません：

- `nsgd`：`adaptation_rate=0.05`、`stats_memory_steps=100`、`plasticity_decay=0.99`、`plasticity_min=0.01`、`plasticity_recovery=1.0`、`knn=12`、`graph_theta=0.1`、`k_attr=1.0`、`k_pulse=2.0`、`k_repulse=0.0`、`max_step_norm=5.0`、`T_push=10`、読み出しは DBSCAN `eps=1.0`、`min_samples=3`、100 ステップごとに再クラスタリング。
- `rg`：上書きなし。`syncmap/SyncMapRG.py` 内の `SyncMapX_RG.DEFAULT_PARAMS` を使います。主な値：`adaptation_rate=0.05`、`level_count=2`（階層グラフでは自動的に深さ + 1 レベルに引き上げられ、例えば DCH1 では 3、IEH では 5 になります。フラットグラフでは上位レベルが無効になります）、`stats_mode=transition`、`stats_memory_steps=300`（レベルごとに `stats_memory_scale=10`）、距離スケールはレベルごとに x3.16、`warmup_steps=2000`、`k_attr=0.07`、`k_repulse=1.0`、`k_macro_repulse=0.1`、読み出しは `physical_mst_gaps`。
- `syncmap`：DBSCAN `eps=3.0`、`min_samples=2`、`space_size=10`。
- `symmetric`：`state_memory=3`、`stochastic_prob=0.3`、`moving_average_window=10000`、DBSCAN `eps=4.5`、`min_samples=2`。
- `tsf`：`adaptation_rate=0.001`、`velocity_decay=0.999`、`mu_pos=6`、`mu_neg=3`、`mu_repulse=2`、読み出しは Ward 法による凝集型クラスタリング（`target_levels=3`、クラスタ数 2-16）。
- 感度分析用プリセット：`rg_shortmem`、`rg_shortmem_l0`、`tsf_longmem`、`tsf_velocity_longmem`。これらは条件を制御した変種であり、別のアルゴリズムではありません。

タスク 2 のデフォルト（`MAIN_CONFIG`）：系列長 5,000 と 100,000、試行 5 回、評価指標 ARI。`--profiles` を指定しない場合、タスク 2 は `syncmap nsgd rg tsf` を、上記のプリセットではなく各クラスのデフォルト値を用いたストラテジーとして実行します。プリセットを使うには `--profiles` を指定してください。

## ベンチマーク

代表的な 8 つのシナリオ、5 つのモデルすべて、100,000 入力ステップ、5 つのシード（0-4）。レベルごとのスコア、潜在マップ、スコア曲線、そして全グラフ・全モデルの GIF を含む完全なレポート：[results/benchmark/REPORT.md](results/benchmark/REPORT.md)。

| シナリオ | 系統 | ノード数 | クラス数（細かい→粗い） | 検証内容 |
|---|---|---:|---|---|
| `fixed5_6` | フラット | 30 | 5 | 基準ケース：大きさの等しい固定順序チャンク |
| `UB_mixed_chunk_20_5_5` | フラット | 30 | 3 | 大きさが大きく異なるチャンク |
| `imbalanced_fixed_300` | フラット | 300 | 14 | 大規模グラフ、不均衡なチャンクサイズ |
| `chain_mixed5_60` | フラット | 300 | 5 | 長い鎖状のチャンク、まばらにしか訪問されない |
| `DCH1` | 階層 | 18 | 6 / 2 | 2 レベルの階層 |
| `IH` | 階層 | 18 | 5 / 2 | 不均衡な階層 |
| `EC` | 階層 | 33 | 15 / 2 | 2 つのグループの下に多数の小さなチャンク |
| `IEH` | 階層 | 18 | 7 / 5 / 4 / 2 | 4 レベルの階層 |

![データセット別・モデル別の ARI](results/benchmark/figures/comparison_ari.png)

ARI、5 シードでの平均 ± 標準偏差（太字 = 最良。階層グラフのスコアは全レベルの平均）：

| シナリオ | SyncMap | Symmetric | TSFMap | NSGD | RG |
|---|---:|---:|---:|---:|---:|
| fixed5_6 | 0.516 ± 0.119 | 0.747 ± 0.141 | 0.788 ± 0.199 | **1.000 ± 0.000** | 0.957 ± 0.033 |
| UB_mixed_chunk_20_5_5 | 0.253 ± 0.170 | **0.594 ± 0.178** | 0.049 ± 0.148 | 0.357 ± 0.064 | 0.592 ± 0.296 |
| imbalanced_fixed_300 | 0.181 ± 0.041 | 0.367 ± 0.135 | 0.012 ± 0.006 | 0.202 ± 0.022 | **0.505 ± 0.090** |
| chain_mixed5_60 | **0.349 ± 0.121** | 0.320 ± 0.089 | 0.022 ± 0.012 | 0.340 ± 0.098 | 0.332 ± 0.079 |
| DCH1 | 0.632 ± 0.003 | 0.643 ± 0.018 | 0.691 ± 0.083 | 0.665 ± 0.033 | **0.795 ± 0.141** |
| IH | 0.221 ± 0.078 | -0.022 ± 0.050 | 0.346 ± 0.311 | **0.924 ± 0.000** | 0.920 ± 0.039 |
| EC | 0.542 ± 0.000 | 0.542 ± 0.000 | 0.281 ± 0.178 | 0.606 ± 0.084 | **0.700 ± 0.121** |
| IEH | 0.341 ± 0.143 | 0.220 ± 0.261 | 0.463 ± 0.082 | **0.654 ± 0.081** | 0.640 ± 0.129 |

所見：

- 大きさの等しいフラットなチャンク（`fixed5_6`）：NSGD はすべてのシードで完璧で、RG もそれに近い値です。
- 不均等・不均衡なチャンク：RG は 300 ノードの不均衡グラフで最良（他のモデルは最大でも 0.367 なのに対し 0.505）で、`UB_mixed_chunk_20_5_5` では Symmetric と同等ですが、シード間のばらつきが大きくなっています。
- 鎖状（`chain_mixed5_60`）：どのモデルも解けていません。SyncMap 系のモデルは 0.32-0.35 前後にとどまり、TSFMap は失敗します。これは既知の鎖状構造での失敗と一致します。
- 階層：`DCH1` と `EC` では、SyncMap、Symmetric、NSGD、RG のいずれも粗いグループを復元します（レベル 2 の ARI 1.0）。差が出るのは細かいレベルで、RG がこれらを上回ります（他が最大でも 0.330 と 0.211 なのに対し、0.589 と 0.399）。TSFMap は逆のパターンを示し、細かいレベルは最良（0.966、0.498）ですが、粗いレベルは最も弱くなっています（0.416、0.064）。全体では両グラフとも RG が最良です。
- 不均衡な階層（`IH`）：NSGD と RG は約 0.92 に達します。TSFMap はシードによってばらつき、Symmetric は失敗します。
- 深い階層（`IEH`）：NSGD と RG は同等です。レベル 1 と 3 では RG、レベル 2 では SyncMap、最も粗いレベルでは NSGD が首位です。
- コスト：100,000 ステップの 1 回の実行に、NSGD が 4-9 秒であるのに対し RG は 16-89 秒かかります（複数の実行を並列に実施）。
- 単一シードのスコア曲線は学習の後半になっても変動し続けるため、表では 5 シードの平均を示しています。

`EC` における最終的な 3 次元配置（シード 0、各モデル自身の座標、真の細かいチャンクで色分け）。SyncMap、Symmetric、NSGD、RG はチャンクを 2 つの分離したグループにまとめますが、TSFMap はチャンクを空間全体に均等に広げるため、粗いレベルが一切現れません：

![EC における 3 次元潜在空間](results/benchmark/figures/latent_EC.png)

`EC` での学習の様子（シード 0。奥行きが分かるようにカメラがゆっくり回転します）：

| SyncMap | Symmetric | TSFMap | NSGD | RG |
|---|---|---|---|---|
| ![EC 上の SyncMap](results/benchmark/gifs/EC__syncmap.gif) | ![EC 上の Symmetric](results/benchmark/gifs/EC__symmetric.gif) | ![EC 上の TSFMap](results/benchmark/gifs/EC__tsf.gif) | ![EC 上の NSGD](results/benchmark/gifs/EC__nsgd.gif) | ![EC 上の RG](results/benchmark/gifs/EC__rg.gif) |

すべてを再実行する（デフォルトの 3 ワーカーで約 40 分。300 ノードのグラフでは各バッチプロセスが数 GB を使うことがあります）か、図とレポートのみを再構築します：

```bash
python tools/benchmark.py run
python tools/benchmark.py report
```

ここでの SyncMap と Symmetric のベースラインは、それぞれの論文にある一定の学習率を使用しています。

## 補足実験

ベンチマークの後に設計した追加実験です。ベンチマークでは NSGD と RG のどちらがどこで勝つのかを判断できず、またその階層は粗いレベルで簡単すぎました。11 グラフ（うち 8 つは生成したもの）、5 つのモデルに再帰をオフにした RG を加えたもの、対応のある 5 つのシード、100,000 ステップで、最後の 5 つのチェックポイントの平均をスコアとします。完全なレポート：[results/supp/REPORT.md](results/supp/REPORT.md)。

- **コミュニティの構成がフラットグラフでの勝者を決めます。** 状態数を 300 に固定すると、100 状態 × 3 コミュニティから 20 状態 × 15 コミュニティまでは、RG がすべてのシードで NSGD を上回ります（ARI で 0.23-0.38 の差）。10 状態 × 30 コミュニティからは NSGD が勝ち、5 状態 × 60 コミュニティでは完璧です。
- **粗いレベルを難しくすると両者に差がつきます。** ウォークが最上位のグループ間をより多く行き来すると、新しい階層グラフでは NSGD が崩れる（0.06-0.10）一方、RG は 0.17-0.42 を保ちます。混合が容易な設定では、2 つのグラフで両者は同等で、DCH1 では RG が、3 レベルのグラフでは NSGD がリードします。
- **効いているのは再帰です。** 再帰なしの RG は、12 の階層設定すべてで完全な RG を下回り（0.05-0.31 の差）、混合が難しいほどその差は大きくなります。
- **読み出しが重要です。** 真のクラスタ数を与えると（oracle-K 診断：正解のクラスタ数 K を既知として切り分ける診断）、12 設定中 11 設定で RG のマップは NSGD と同等以上になります。正式な MST カットによって、どのモデルも ARI が 0.07-0.40 下がります。

![300 状態におけるコミュニティ数と ARI の関係](results/supp/figures/orientation_sweep.png)

![粗いレベルの混合度とレベル平均 ARI の関係](results/supp/figures/hierarchy_difficulty.png)

```bash
python tools/make_supp_datasets.py --stats
python tools/supp_run.py run --workers 3
python tools/supp_run.py showcase
python tools/supp_report.py
```

## 評価

- すべての手法は、その時点の座標に対する同一の教師なし読み出しによってスコア付けされます。正解ラベルはスコア計算にのみ使われ、レベルの選択には一切使われません。
- フラットグラフ：標準化した座標に対する DBSCAN（eps = 1.25 × k 距離の中央値、`min_samples=2`）。その分割の ARI / NMI を求めます。
- 階層グラフ：ユークリッド最小全域木を、隣接するエッジ間の大きなギャップ（比 ≥ 1.05）で切断します。必要なレベルごとに 1 回ずつ切断します。正式な ARI は、必要な深さにわたる `cluster_level1` ... `cluster_levelN` の平均で、欠けているレベルのスコアは 0 とします。NMI は固定レベルの診断指標のままとし、コーフェネティック相関とレベルごとのスコアは補足的な情報を与えます。
- 診断専用：`oracle_cluster_levelN_ari_score` は、同じ座標を真のクラスタ数で Ward クラスタリングにより切り分けます。oracle スコアが高く正式スコアが低い場合、マップは構造を保持しているのに読み出しがそれを取りこぼしていることを意味します。
- `--seed` を指定すると、入力ストリームは環境専用の乱数ストリームから生成されるため、同じシードではすべてのモデルが同じ系列を見ることになります。

## リポジトリ構成

- `main.py`：コマンドラインのエントリーポイントとタスクの振り分け。
- `unified_experiments.py`：モデルと環境の構築、学習ループ、評価、一括比較。
- `live.py`：対話型ビューアのバックエンド（タスク 1）。
- `syncmap/`：アルゴリズムの実装。
- `test/`：実行時に使うグラフウォーク環境と階層環境、および対象を絞った単体テスト。
- `tools/`：プロファイル（`main_config.py`）、読み出しと評価指標、プロット、CSV の集計、ベンチマーク（`benchmark.py`）、補足実験（`make_supp_datasets.py`、`supp_run.py`、`supp_report.py`）、プレゼンテーション用の図（`deck_figures.py`）。
- `datasets/`：合成 `.dot` グラフ。階層グラフは `datasets/hierarchy/`、補足実験用に生成したグラフは `datasets/supp/` にあります。
- `results/benchmark/`：ベンチマークのレポート、比較 CSV、図、GIF。`results/supp/`：補足実験のレポート、比較 CSV、図。その他の `results/` 内のフォルダは実行時の出力であり、バージョン管理の対象外です。
- `docs/`：ドキュメントマップ、プレゼンテーション、設計リファレンス、CLI リファレンス、出力先、計画、制限事項。

## ドキュメント

- [ドキュメントマップ](docs/README.md)
- [プレゼンテーション](docs/presentation/index.html)（HTML スライド。単一ファイル版と PDF は [リリースページ](https://github.com/Cyletix/SyncMapRGCI/releases) にあります）
- [ベンチマークレポート](results/benchmark/REPORT.md) と [補足実験レポート](results/supp/REPORT.md)
- [設計と評価](docs/DESIGN_REFERENCE.md)
- [CLI リファレンス](docs/CLI_REFERENCE.md)
- [リポジトリ構成](docs/PROJECT_STRUCTURE.md)
- [出力先](docs/OUTPUT_GOVERNANCE.md)
- [未解決の課題](docs/PLAN.md) と [既知の制限事項](docs/ISSUES_LOG.md)

## 制限事項

- 入力ステップ数が同じでも、手法間で計算量やメモリ使用量が等しいわけではありません。
- ベンチマークが扱うのは 8 つの合成グラフ、補足実験は 11 グラフ（うち 8 つは補足実験のために生成）です。これらは代表的なケースや条件を制御したケースでの挙動を示すものであり、一般的な優位性を示すものではありません。
- 階層の読み出しは、単一チェックポイントにおける MST のギャップカットです。マップ上で明確に分離されている粗いグループを見逃すことがあるため、粗いレベルのスコアが低いことを表現の失敗と解釈する前に、oracle-K の列と比較してください。
- RG のデフォルトのレベルスケジューラでは深いレベルのクロックがエイリアシングを起こすため、L1 より上のレベルは 100,000 ステップ以内ではほとんどエビデンスを受け取れません（[詳細](docs/ISSUES_LOG.md)）。
- 階層データセットは、パス（`datasets/hierarchy/`）または `.dot` の内容（内部ノードや多レベルラベル）によって判定されます。`--env-mode` でこの判定を上書きできます。
- CI は探索的な段階にあり、検証済みの因果推論システムはまだ確立されていません。
