import requests
import json
import os

while 1:
    #f-graphictags.tagnameに取得したいタグ名を入れてください。（例えば以下は地図資料）
    rawdata=requests.get("https://lab.ndl.go.jp/dl/api/illustration/randomwithfacet?size=10&f-graphictags.tagname=graphic_map",
                         headers={"content-type": "application/json"})
    for item in rawdata.json():
        print(item)
        imgname=item["id"]+".jpg"
        imgurl="https://www.dl.ndl.go.jp/api/iiif/{}/R{:07}/pct:{},{},{},{}/,256/0/default.jpg".format(
            item["pid"],item["page"],item["x"],item["y"],item["w"],item["h"])
        rawimg=requests.get(imgurl).content
        if len(item['graphictags'])==0:
            os.makedirs(os.path.join("output", "unknown"), exist_ok=True)
            with open(os.path.join("output", "unknown", imgname), "wb") as fout:
                fout.write(rawimg)
            continue
        os.makedirs("output",exist_ok=True)
        with open(os.path.join("output",imgname), "wb") as fout:
            fout.write(rawimg)
    break #繰り返し取得する場合はsleep等をいれてください。



