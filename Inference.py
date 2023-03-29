import torch
import torch.nn.functional as F
from torchvision import transforms
import base64
from io import BytesIO
from PIL import Image
import onnxruntime
import numpy as np


class Inference:
    def __init__(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(224, 224))])
        self.ort_session = onnxruntime.InferenceSession("emotion_3080.onnx")

    def forward(self, img):
        im_bytes = base64.b64decode(img)   # im_bytes is a binary image
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        img = Image.open(im_file)   # img is now PIL Image object

        img_transform = np.array(self.transform(img))
        img_transform = img_transform.reshape(1,3,224,224)
        ort_inputs = {self.ort_session.get_inputs()[0].name: np.array(img_transform)}
        ort_outs = self.ort_session.run(None, ort_inputs)
        sm = torch.tensor(ort_outs)
        sm = sm.reshape(7,-1)
        #print("output", sm, sm.shape)
        ort_outs = F.softmax(sm, dim=0)
        #print("RESULTS", ort_outs)
        return ort_outs