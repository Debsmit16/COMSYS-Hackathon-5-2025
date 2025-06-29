import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import random
from .preprocessing import AdversarialPreprocessor
from .augmentation import AdvancedAugmentation

class COMSYSDataset:
    def __init__(self, config):
        self.config = config
        self.preprocessor = AdversarialPreprocessor(size=tuple(config["image_size"]))
        self.augmentation = AdvancedAugmentation()
        self.batch_size = config["batch_size"]

    def load_gender_dataset(self, data_path, is_training=True):
        data_path = Path(data_path)
        images, labels = [], []
        for label, subdir in enumerate(["male","female"]):
            sub_path = data_path/subdir
            if sub_path.exists():
                for img_path in sub_path.glob("*.*"):
                    img = cv2.imread(str(img_path))
                    if img is None: continue
                    proc = self.preprocessor.preprocess(img)
                    if is_training: proc = self.augmentation.apply_augmentation(proc)
                    images.append(proc); labels.append(label)
        images = np.array(images,dtype=np.float32); labels = np.array(labels,dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((images,labels))
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def load_face_recognition_dataset(self, data_path, is_training=True):
        data_path = Path(data_path)
        identity_images = {}
        for d in data_path.iterdir():
            if not d.is_dir() or d.name in ["male","female","distorted"]: continue
            imgs=[]
            for img_path in d.glob("*.*"):
                img=cv2.imread(str(img_path))
                if img is None: continue
                imgs.append(self.preprocessor.preprocess(img))
            if len(imgs)>=2: identity_images[d.name]=imgs
        if is_training:
            return self._generate_triplets(identity_images)
        else:
            return self._create_validation_pairs(identity_images)

    def _generate_triplets(self, identity_images, num_triplets=1000):
        anchors,positives,negatives=[],[],[]
        ids=list(identity_images.keys())
        for _ in range(num_triplets):
            aid=random.choice(ids); posid=aid
            negid=random.choice([i for i in ids if i!=aid])
            a,p = random.sample(identity_images[aid],2)
            n = random.choice(identity_images[negid])
            anchors.append(self.augmentation.apply_augmentation(a))
            positives.append(self.augmentation.apply_augmentation(p))
            negatives.append(self.augmentation.apply_augmentation(n))
        arr=lambda l:np.array(l,dtype=np.float32)
        ds=tf.data.Dataset.from_tensor_slices({"anchor":arr(anchors),"positive":arr(positives),"negative":arr(negatives)})
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def _create_validation_pairs(self, identity_images):
        pairs,labels=[],[]
        ids=list(identity_images.keys())
        for i in ids:
            imgs=identity_images[i]
            for a in range(len(imgs)):
                for b in range(a+1,len(imgs)):
                    pairs.append((imgs[a],imgs[b])); labels.append(1)
        for _ in labels:
            i,j=random.sample(ids,2)
            pairs.append((random.choice(identity_images[i]),random.choice(identity_images[j]))); labels.append(0)
        arr=lambda l:np.array([x for x,y in l],dtype=np.float32)
        imgs1=arr(pairs); imgs2=np.array([y for x,y in pairs],dtype=np.float32)
        ds=tf.data.Dataset.from_tensor_slices({"img1":imgs1,"img2":imgs2,"label":np.array(labels,dtype=np.float32)})
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

def make_tf_dataset(config, task="gender"):
    loader=COMSYSDataset(config)
    if task=="gender":
        return loader.load_gender_dataset(config["train_path"]), loader.load_gender_dataset(config["val_path"],is_training=False)
    elif task=="face":
        return loader.load_face_recognition_dataset(config["train_path"]), loader.load_face_recognition_dataset(config["val_path"],is_training=False)
    else:
        raise ValueError("Unknown task:"+task)