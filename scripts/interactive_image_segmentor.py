import gc
import gradio as gr
from fastapi import FastAPI, Body,Header,status
from gradio import components,Blocks,Row
import json
from PIL import Image
import torchvision.transforms as transforms
import torch
from pathlib import Path
import os
import requests
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
from modules.safe import unsafe_torch_load, load
from modules.devices import device, torch_gc, cpu

from modules.processing import process_images
import modules.scripts as scripts

UNIT_DEBUG=False
def import_or_install(package,pip_name=None):
    import importlib
    import subprocess
    if pip_name is None:
        pip_name=package
    try:
        importlib.import_module(package)
        print(f"{package} is already installed")
    except ImportError:
        print(f"{package} is not installed, installing now...")
        subprocess.call(['pip', 'install', package])
        print(f"{package} has been installed")

import_or_install("segment_anything","git+https://github.com/facebookresearch/segment-anything.git")

class InteractiveImageSegmentor:
    def download_file_if_not_exists(file_url, file_name):
        if not os.path.isfile(file_name):
            response = requests.get(file_url)
            if response.status_code == 200:
                with open(file_name, 'wb') as file:
                    file.write(response.content)
                print("File downloaded successfully!")
            else:
                print("Failed to download the file.")

    def load_model(self,model_choice="sam_vit_b"):
        sam_checkpoint=f"{model_choice}.pth"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if model_choice=="sam_vit_b":InteractiveImageSegmentor.download_file_if_not_exists("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",sam_checkpoint)
        elif model_choice=="sam_vit_l":InteractiveImageSegmentor.download_file_if_not_exists("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",sam_checkpoint)
        elif model_choice=="sam_vit_h":InteractiveImageSegmentor.download_file_if_not_exists("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",sam_checkpoint)
        model_type=model_choice.replace("sam_","")
        if model_type not in sam_model_registry:
            model_type="default"
        print(f"Loading model {model_type} from {sam_checkpoint}")
        torch.load = unsafe_torch_load
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)
        torch.load = load
    def clear_sam_cache(self):
        self.sam.unload_model()
        gc.collect()
        torch_gc()

    def mask2image_multi(self,mask:torch.Tensor):
        # print(mask.shape)
        if mask.dim() == 3 and mask.shape[-1] == 3:
            mask = mask.permute(2, 0, 1)
        elif mask.dim() == 3 and mask.shape[0] == 3:
            pass
        else:
            print(mask.shape)
            raise ValueError("Mask tensor has an unexpected shape.")
        color = torch.Tensor([255/255, 155/255, 114/255, 0.6]).to(self.device)
        binary_mask = mask[0, :, :]
        h, w = binary_mask.shape
        mask_image = binary_mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return mask_image.permute(2, 0, 1)
    def mask2image(self, mask: torch.Tensor):
        if mask.dim() == 3 and mask.shape[0] == 1:
            binary_mask = mask.squeeze(0)
        elif mask.dim() == 2: 
            binary_mask = mask
        else:
            print(mask.shape)
            raise ValueError("Mask tensor has an unexpected shape.")
        h, w = binary_mask.shape
        rgb_image = binary_mask.repeat(3, 1, 1)
        alpha_channel = torch.full((1, h, w), 0.6).to(self.device)
        rgba_image = torch.cat((rgb_image, alpha_channel), dim=0)
        color = torch.Tensor([255/255, 155/255, 114/255]).to(self.device).reshape(3, 1, 1)
        return rgba_image
    def preview_segment(self,image:Image,points:list[list[float]]=None,bbox=None,labels:list[int]=None):
        pil_2_tensor = transforms.PILToTensor()
        rgba_image = image.convert("RGBA")
        image_tensor = pil_2_tensor(rgba_image).cuda()
        result_tensor=image_tensor.clone()
        mask_tensor=self.segment(points,bbox,labels)
        mask_image_tensor=self.mask2image(mask_tensor)
        mask_image=transforms.ToPILImage()(mask_image_tensor)
        result_image:Image = transforms.ToPILImage()(result_tensor)
        result_image=Image.alpha_composite(result_image,mask_image)
        return result_image
    def segment(self,points:list[list[float]]=None,bbox=None,labels:list[int]=None)->torch.Tensor:
        if len(points)==0:points=None
        if len(labels)==0:labels=None
        if len(bbox)==0:bbox=None
        if points is not None:points = torch.Tensor(np.array(points)).to(self.device).unsqueeze(0)
        if labels is not None:labels = torch.Tensor(np.array(labels)).to(self.device).unsqueeze(0)
        if bbox is not None:bbox = torch.Tensor(np.array(bbox)).to(self.device)
        print(points,labels,bbox)
        masks, scores, logits = self.predictor.predict_torch(
            point_coords=points,
            point_labels=labels,
            boxes=bbox,
            multimask_output=False,
        )
        return masks[0]
    
    def remove_selected(self,image:Image,points:list[list[float]]=None,boxes=None,labels:list[int]=None):
        pil_2_tensor = transforms.PILToTensor()
        rgba_image = image.convert("RGBA")
        image_tensor = pil_2_tensor(rgba_image).cuda()
        mask_tensor = image_segmentor.segment(points=points,bbox=boxes,labels=labels)
        result_tensor=image_tensor*(1-mask_tensor)
        result_image:Image = transforms.ToPILImage()(result_tensor)
        return result_image
    def remove_unselected(self,image:Image,points:list[list[float]]=None,boxes=None,labels:list[int]=None):
        pil_2_tensor = transforms.PILToTensor()
        rgba_image = image.convert("RGBA")
        image_tensor = pil_2_tensor(rgba_image).cuda()
        mask_tensor = image_segmentor.segment(points=points,bbox=boxes,labels=labels)
        print(image_tensor.shape,mask_tensor.shape)
        result_tensor=image_tensor*mask_tensor
        result_image:Image = transforms.ToPILImage()(result_tensor)
        return result_image
    pass

def reset_image(image:Image):
    global image_segmentor
    if image_segmentor is None:
        image_segmentor=InteractiveImageSegmentor()
        image_segmentor.load_model()
    image_segmentor.predictor.reset_image()
    image_array = np.array(image)
    image_segmentor.predictor.set_image(image_array)
    return image

def on_image_changed(image:Image):
    global points,labels,box_cache,boxes
    points=[]
    labels=[]
    boxes=[]
    box_cache=[]
    reset_image(image)
    return image

def on_image_clicked(image:Image,choice,input_type,event_data:gr.SelectData):
    global box_cache,boxes,points,labels
    if isinstance(choice,str):
        if choice=="Select":choice=1
        elif choice=="Deselect":choice=0
    if input_type=="Point":
        points.append(event_data.index)
        labels.append(choice)
        return image_segmentor.preview_segment(image,points=points,bbox=boxes,labels=labels)
    elif input_type=="Box":
        box_cache.extend(event_data.index)
        if len(box_cache)==4:
            boxes.append(box_cache)
            box_cache=[]
            return image_segmentor.preview_segment(image,points=points,bbox=boxes,labels=labels)
    return image

def on_remove_btn_clicked(image:Image,remove_type:str):
    global points,labels,box_cache,boxes
    if remove_type=="Selected":
        return image_segmentor.remove_selected(image,points=points,boxes=boxes,labels=labels)
    elif remove_type=="Unselected":
        return image_segmentor.remove_unselected(image,points=points,boxes=boxes,labels=labels)
    return image

class Script(scripts.Script):
    def title(self):
        return "Interactive Image Segmentor"
    def show(self, is_img2img):
        return is_img2img
    def ui(self, is_img2img):
        if not is_img2img: return
        with Blocks():
            with Row(equal_height=True):
                choice=components.Radio(choices=["Select","Deselect"],value="Select",label="Selection Type")
                input_type=components.Radio(choices=["Point","Box"],value="Point",label="Input Type")
                remove_type=components.Radio(choices=["Selected","Unselected"],value="Selected",label="Remove Type")
            with Row(equal_height=True):
                image=components.Image(type="pil",interactive=True,image_mode="RGB")
                resulting_image=components.Image(type="pil",image_mode="RGBA")
                image.change(on_image_changed,inputs=[image],outputs=[resulting_image])
                image.select(on_image_clicked,inputs=[image,choice,input_type],outputs=[resulting_image])
            with Row(equal_height=True):
                remove_btn = components.Button(value="Preview Remove Effect")
                remove_btn.click(on_remove_btn_clicked,inputs=[image,remove_type],outputs=[resulting_image])
                pass
        return [image,points,labels,boxes]

    def run(self,p,image,points,labels,boxes):
        if image is None:
            image=p.init_images[0]
        image_segmentor.predictor.set_image(np.array(image))
        mask=image_segmentor.predictor.predict_torch(points,labels,boxes)
        p.image_mask=mask
        proc = process_images(p)
        proc.images.append(mask)
        return proc
    pass

def interactive_image_segmentor_api(_: Blocks, app: FastAPI):
    @app.post("/figma/interactive_image_segmentor/upload_image")
    async def upload_image(image_str:str = Body(...)):
        import base64
        import io
        image_bytes = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_bytes),formats=["PNG"])
        image_segmentor.predictor.reset_image()
        image_segmentor.predictor.set_image(np.array(image))
        return image
    @app.post("/figma/interactive_image_segmentor/image_x_mask")
    async def remove_selected(image_str:str = Body(...),points:list[list[float]]=Body(...),\
            boxes:list[list[float]]=Body(...),labels:list[int]=Body(...), remove_type:bool=Body(...)):
        import base64
        import io
        image_bytes = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_bytes),formats=["PNG"])
        if remove_type=="Selected":
            image= image_segmentor.remove_selected(image,points=points,boxes=boxes,labels=labels)
        elif remove_type=="Unselected":
            image= image_segmentor.remove_unselected(image,points=points,boxes=boxes,labels=labels)
        return image
    pass

points=[]
labels=[]
box_cache:list=[]
boxes=[]

image_segmentor=None

# with Blocks() as demo:
#     with Row(equal_height=True):
#         choice=components.Radio(label="Choice",choices=["Select","Deselect"],value="Select")
#         input_type=components.Radio(label="Input Type",choices=["Point","Box"],value="Point")
#         remove_type=components.Radio(choices=["Selected","Unselected"],value="Selected")
#     with Row(equal_height=True):
#         image=components.Image(type="pil",interactive=True,image_mode="RGB")
#         resulting_image=components.Image(type="pil",image_mode="RGBA")
#         image.change(on_image_changed,inputs=[image],outputs=[resulting_image])
#         image.select(on_image_clicked,inputs=[image,choice,input_type],outputs=[resulting_image])
#     with Row(equal_height=True):
#         remove_btn = components.Button(value="Preview Remove Effect")
#         remove_btn.click(on_remove_btn_clicked,inputs=[image,remove_type],outputs=[resulting_image])
#         pass
# demo.queue()
# demo.launch()
try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(interactive_image_segmentor_api)
except:
    pass
pass