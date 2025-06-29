import cv2, numpy as np

class AdversarialPreprocessor:
    def __init__(self, size=(224,224)):
        self.size = size
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def _clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _gamma(self, img, g):
        table = (np.array([(i/255.0)**(1/g)*255 for i in range(256)])).astype("uint8")
        return cv2.LUT(img, table)

    def preprocess(self, img):
        face = self._align(img)
        face = cv2.bilateralFilter(face,9,75,75)
        face = self._clahe(face)
        bright = np.mean(face)
        gamma = 0.7 if bright<85 else 1.3 if bright>170 else 1.0
        face = self._gamma(face, gamma)
        face = cv2.resize(face, self.size, interpolation=cv2.INTER_LANCZOS4)
        return face.astype("float32")/255.

    def _align(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(g,1.1,4)
        if len(faces)==0: return img
        x,y,w,h = max(faces, key=lambda b: b[2]*b[3])
        pad = int(0.2*min(w,h))
        x1,y1 = max(0,x-pad), max(0,y-pad)
        x2,y2 = min(img.shape[1],x+w+pad), min(img.shape[0],y+h+pad)
        return img[y1:y2, x1:x2]