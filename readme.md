# tagestimatemodel
NDL-ImageLabelデータセット
(https://github.com/ndl-lab/imagetagdataset)
を利用して、資料画像に付与するタグを推定するモデルを公開しています。

## 1.概要
NDL-ImageLabelを学習データとして、マルチラベル画像分類（1画像に対して複数のラベルを付与するタスク）を行う学習済モデル、推論ソースコード、学習用ソースコード及び、データセットを拡張するための画像取得スクリプト（次世代デジタルライブラリーAPIを利用）を公開しています。
本リポジトリで公開しているソースコード内のモデルにはEfficientNetB0を利用しています。

## 2.公開しているモデルについて
NDL-ImageLabelに加えて、一般公開していないpicture_personを含めたデータで学習したモデルについて、次のURLから公開しています。

https://lab.ndl.go.jp/dataset/tagestimatemodel/weights_ndllabelimage.hdf5

これは、次世代デジタルライブラリーにおいてタグの自動付与に用いているモデルと同じものです。

性能の目安として、NDL-ImageLabelのうちランダムな1割の画像をvalidation dataとした場合、validation scoreはmacro-F1 scoreで0.89～0.91程度になります。


推論のコード例についてはmultiinference.pyを参照してください。

multiinference.pyと同じ階層にweights_ndllabelimage.hdf5を配置して実行すると、inputフォルダ内の資料について、タグの自動付与が行われた結果がoutputフォルダにjsonとして出力されます。

NDl-ImageLabelのタグはタグ同士が緩やかな包含関係（例：picture_landmarkとpicture_outdoor）を持つように設計しているため、下表のように、1つのタグに1~3つのラベルを対応させて学習しています。

  |NDL-ImageLabelのラベル名| モデル学習時に付与したラベル
  |---------------------|----------
  |graphic_map          |graphic,graphic_map
  |graphic_graph        |graphic,graphic_graph
  |graphic_illustcolor  |graphic,graphic_illustcolor
  |graphic_illust       |graphic,graphic_illust 
  |picture_landmark     |picture,picture_outdoor,picture_landmark
  |picture_outdoor      |picture,picture_outdoor
  |picture_object       |picture,picture_object
  |picture_indoor       |picture,picture_indoor
  |stamp                |stamp
  |(picture_person)     |picture,picture_person


## 3.モデルの再学習について
datasetpdmフォルダ内にNDL-ImageLabelを展開するか、4で取得・分類したオリジナルな画像ファイルを、クラスごとにフォルダに分けて配置してください。NDL-ImageLabelを利用する場合は、
picture_personフォルダは存在しないので自分で取得する必要があります。
配置後、multitrain_efficientnet.pyを実行するとweights_pdm_multiフォルダ内に学習済モデルが出力されます。


## 4. 学習データを追加するには

次世代デジタルライブラリーのAPIを利用すると、任意のタグについて、著作権保護期間満了資料から切り出された画像の情報を取得できます。例えば以下の例では、graphic_map（地図）のタグが付与された画像の情報が10画像分得られます。
```
curl https://lab.ndl.go.jp/dl/api/illustration/randomwithfacet?size=10&f-graphictags.tagname=graphic_map
```
この画像の情報は元の資料画像上の座標情報等を含んでいます。
画像ファイルとして取得したい際には、国立国会図書館デジタルコレクションの提供しているIIIF Image APIのURLを組み立てる必要があります。(参考:https://iiif.io/api/image/2.1/#region)

具体的には、

https://www.dl.ndl.go.jp/api/iiif/(資料のPID)/R(0埋め7桁のコマ番号)/pct:(正規化されたx座標),(正規化されたy座標),(正規化された幅),(正規化された高さ)/(取得したいサイズ)/0/default.jpg

となります。

pythonの場合の画像取得のサンプルコードをnewdataclawer.pyに掲載しています。

今回利用しているモデルは完璧ではないので、上記の手順で取得した画像には誤分類が含まれるはずです。
また、例えば自然の風景（山や森）を分類したいといった、NDL-ImageLabelで設定しているタグより細かい分類のニーズもあると思います。

NDL-ImageLabelや本リポジトリで公開しているモデルはあくまでスタートラインです。
どうぞ次世代デジタルライブラリーのAPIを利用して自分だけのデータセットを作り、
デジタルアーカイブの可能性を広げる機械学習モデルを開発してください。


## 5. 参考文献
青池亨. 図書館における機械学習技術の実践的適用について―次世代デジタルライブラリーの機能改修及び新たなデータセットの公開を中心に. じんもんこん2021.



