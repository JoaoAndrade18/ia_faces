import sys, os, distutils.core
# dist = distutils.core.run_setup("../detectron2/setup.py")
sys.path.insert(0, os.path.abspath('../detectron2'))

import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# register dataset

from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog

register_coco_instances('train_dataset', {}, '../dataset/faces-metadata.json', '../dataset/train/')

dataset_metadata_train = MetadataCatalog.get('train_dataset')
dataset_dict_train = DatasetCatalog.get('train_dataset')

import os
import cv2
from tkinter import Tk, Button, Label, Text, Scrollbar, filedialog, PhotoImage
from PIL import Image, ImageTk
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog

class PeopleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("People Detection App")
        self.root.geometry("400x500")
        self.root.resizable(False, False)

        self.image_label = Label(self.root)
        self.image_label.pack(pady=10)

        # Substitua "path/to/upload_icon.png" e "path/to/detect_icon.png" pelos caminhos reais dos seus ícones.
        self.upload_icon = PhotoImage(file="./images/upload-button.png")
        self.detect_icon = PhotoImage(file="./images/detect-button.png")

        self.upload_button = Button(self.root, text="Upload Image", command=self.upload_image, image=self.upload_icon, compound="left", bg="#4CAF50", fg="white")
        self.upload_button.pack(pady=5)

        self.detect_button = Button(self.root, text="Detect People", command=self.detect_people, image=self.detect_icon, compound="left", bg="#008CBA", fg="white")
        self.detect_button.pack(pady=5)

        self.output_text = Text(self.root, wrap="word", height=5, width=50, bg="#f0f0f0", fg="#333", bd=2, relief="solid", font=("Helvetica", 10))
        self.output_text.pack(pady=10)

        self.status_label = Label(self.root, text="", bd=1, relief="sunken", anchor="w")
        self.status_label.pack(side="bottom", fill="x")

        # Load the model
        self.cfg = get_cfg()
        self.cfg.merge_from_file("../detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.cfg.MODEL.DEVICE = 'cpu'
        self.cfg.MODEL.WEIGHTS = os.path.join("./model_final.pth")
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 13
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
        self.predictor = DefaultPredictor(self.cfg)

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = Image.open(file_path)
            image = image.resize((300, 300), Image.LANCZOS)
            self.image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image)
            self.image_label.image = self.image
            self.file_path = file_path

    def detect_people(self):
        if hasattr(self, 'file_path'):
            im = cv2.imread(self.file_path)
            outputs = self.predictor(im)

            class_names = MetadataCatalog.get('train_dataset').thing_classes

            peoples = []
            for i in range(len(outputs["instances"])):
                class_id = outputs["instances"].pred_classes[i].cpu().numpy()
                class_name = class_names[class_id]
                peoples.append(class_name)

            peoples = list(set(peoples))
            self.output_text.delete(1.0, "end")
            self.output_text.insert("end", "People detected: " + ", ".join(peoples))
        else:
            self.output_text.delete(1.0, "end")
            self.output_text.insert("end", "Please upload an image first.")

if __name__ == "__main__":
    root = Tk()
    app = PeopleDetectionApp(root)
    root.mainloop()