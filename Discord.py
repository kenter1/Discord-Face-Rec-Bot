import discord
import os
import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
import requests
from io import BytesIO

from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import prepare_facebank
import numpy as np

def getImage(img_url):
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    img = img[:, :, :3]
    img = Image.fromarray(img)

    return img

def recognize(img):
    bboxes, faces = mtcnn.align_multi(img, conf.face_limit, conf.min_face_size)
    bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
    bboxes = bboxes.astype(int)
    bboxes = bboxes + [-1, -1, 1, 1]  # personal choice
    results, score = learner.infer(conf, faces, targets, "store_true")

    return results, score

def updateDatabase():
    return prepare_facebank(conf, learner.model, mtcnn, tta="store_true")

class MyClient(discord.Client):
    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):
        global targets, names
        print("message: {}".format(message.content))

        if message.author.id == self.user.id:
            return

        elif(".model" in message.content):
            await message.channel.send("Architecture: {}\n Paper: {}\n Github link: {}\n Face Recognition implementation and Pretrained Model From: {}"
                .format("Arcface",
                        "https://arxiv.org/abs/1801.07698",
                        "https://github.com/deepinsight/insightface",
                        "https://github.com/TreB1eN/InsightFace_Pytorch"))

        elif(".rec" in message.content):
            if(len(message.attachments) > 0):
                img_url = message.attachments[0].url
                img = getImage(img_url)
                try:
                    results, scores = recognize(img)
                except:
                    await message.channel.send("Image parsing error")
                    return

                await message.channel.send("Found:")
                for idx in range(len(results)):
                    await message.channel.send("{} {:.2f}%".format(names[results[idx] + 1], scores[idx] * 100))

            else:
                await message.channel.send("No Image Attached")

        elif(".add" in message.content):
            label = message.content[5:]
            if (label == ""):
                await message.channel.send(".add (Name)")
                return

            if (len(message.attachments) > 0):
                img_url = message.attachments[0].url
                img = getImage(img_url)
                try:
                    results, scores = recognize(img)
                except:
                    await message.channel.send("Image parsing error")
                    return

                if(names[results[0] + 1] != "Unknown"):
                    await message.channel.send("Person already exists!: {}".format(names[results[0] + 1]))
                    return
                else:
                    path = r"E:\FaceRecognition\InsightFace_Pytorch-master\data\facebank"
                    newpath = r"{}\{}".format(path, label)
                    try:
                        os.mkdir(newpath)
                    except OSError:
                        print("Already Exists")
                    else:
                        print("New Face %s " % path)

                    image_path = r"{}\{}".format(newpath, message.attachments[0].filename)
                    img.save(image_path)

                    targets, names = updateDatabase()
                    await message.channel.send("Added: {}".format(label))
            else:
                await message.channel.send("No Image Attached")





if(__name__ == "__main__"):
    conf = get_config(False)
    mtcnn = MTCNN()
    print('mtcnn loaded')

    learner = face_learner(conf, True)
    learner.threshold = 1.54
    learner.load_state(conf, 'ir_se50.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    targets, names = updateDatabase()
    print('facebank updated')

    client = MyClient()
    client.run("omitted")
