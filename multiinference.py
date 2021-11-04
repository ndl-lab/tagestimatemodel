from tensorflow.keras.utils import Sequence,to_categorical
from tensorflow.keras.applications import Xception,ResNet101V2
from efficientnet.keras import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
import glob
import json
import numpy as np
import os
import shutil
import json
from argparse import ArgumentParser
base_model =EfficientNetB0(
    include_top = False,
    weights = None,
    input_shape = None
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='linear', activity_regularizer=None)(x)
x = Dense(512, activation = 'relu')(x)
predictions = Dense(11, activation = 'sigmoid')(x)
model = Model(inputs = base_model.input, outputs = predictions)
print(model.summary())

classdic={0:"graphic",1:"picture",2:"graphic_graph",3:"graphic_map",4:"graphic_illustcolor",
        5:"picture_indoor",6:"picture_object",7:"picture_person",8:"picture_outdoor",9:"picture_landmark",10:"stamp"}
weights_path = 'val_loss0.072.hdf5'
model.load_weights(weights_path)
import random
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inputpath', default="input",help='inputpath(default: ./input)')
    parser.add_argument('--outputpath', default="output", help='outputpath(default: ./output)')
    args = parser.parse_args()
    image_dirpaths = glob.glob(os.path.join(args.inputpath,"*"))
    partsize=len(image_dirpaths)//4
    for datadir in image_dirpaths:
        img_predict = []
        img_pathlist=[]
        pid=os.path.basename(datadir)
        resjson={}
        for index,image_name in enumerate(os.listdir(datadir)):
            try:
                img = cv2.imread(os.path.join(datadir, image_name))
                img = cv2.resize(img,(224,224))/255.0
                img_predict.append(img)
                img_pathlist.append(os.path.join(datadir, image_name))
            except Exception as e:
                print(e)
                pass
            if index%64==0 or index==len(os.listdir(datadir))-1:
                if len(img_predict)==0:
                    continue
                img_predict = np.asarray(img_predict)
                result_predict = model.predict(img_predict)
                for i in range(len(img_predict)):
                    taglist=[]
                    for cindex in range(len(classdic)):
                        if result_predict[i][cindex]>0.4:
                            taglist.append({"tag":classdic[cindex],"confidence":round(result_predict[i][cindex].item(),3)})
                    fileid = os.path.basename(img_pathlist[i])[:-4]
                    resjson[fileid]=taglist
                img_predict=[]
                img_pathlist=[]
        os.makedirs(args.outputpath, exist_ok=True)
        with open(os.path.join(args.outputpath,pid+".json"),"w",encoding="utf-8") as f:
            json.dump(resjson,f)

