# 外部ソフトウェアと記録の出典

このリポジトリは、選定した自分のロボット実行記録と、それを扱う最小限のコードを収録します。元プロジェクトの Git 履歴、講座配布資料、競技の評価システム、モデル重み、外部ライブラリのソースやシーンアセットは同梱しません。

## 外部依存

| 依存先 | 役割 | 参照先 |
|---|---|---|
| OpenVLA-OFT | 画像・言語・ロボット状態からの行動推論 | https://github.com/moojink/openvla-oft |
| LIBERO-plus | 録画時に使ったシミュレータと公開タスク | https://github.com/sylvestf/LIBERO-plus |
| LIBERO | 基盤となるロボットシミュレーションとアセット | https://github.com/Lifelong-Robot-Learning/LIBERO |
| LIBERO-plus assets | 拡張シーンのアセット | https://huggingface.co/datasets/Sylvest/LIBERO-plus |
| NumPy | 動作配列の読み取り・検証 | https://numpy.org/ |
| FFmpeg / imageio-ffmpeg | 映像の連結・画面配置・出力 | https://ffmpeg.org/ / https://github.com/imageio/imageio-ffmpeg |
| Pillow | 検査用画像の作成 | https://python-pillow.github.io/ |

外部依存を取得する際は、提供元のライセンス・利用条件に従ってください。このリポジトリに包括的な第三者資産のライセンスを付与するものではありません。特に、公開リポジトリで閲覧可能であることと、ソース・アセットの再配布が許可されていることは別です。ここでは上流参照のみを残します。

## 映像と推論の関係

`media/source/` は、追加学習済み OpenVLA-OFT `mixed_000500` による既存のロボット実行映像の選抜です。生成映像や手動操作の映像ではありません。`data/episodes/` の行動は、その実行時に記録されたものです。

保存行動の再生は、新たな AI 推論ではありません。新規推論には、外部の推論コードと別途用意したモデル重みが必要です。同じ名前の公開ベースモデルを用意するだけでは、今回の追加学習済みモデルと同じ動作になることを保証しません。

各記録の指示・実行条件・成功判定・ハッシュは `data/manifest.json` と `docs/provenance.md` に記載します。
