from PIL import Image
import numpy as np

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images, fix_seed
from modules.shared import opts, cmd_opts, state
import modules.scripts as scripts
from fastapi import FastAPI, Body,Header,status
from gradio import components,Blocks,Row


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

import_or_install("rembg","rembg[gpu]")
sessions=dict()

def remove_background(image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,alpha_matting_erode_size,\
                      session_name,only_mask,post_process_mask):
    from rembg import remove, new_session
    if session_name not in sessions:
        sessions[session_name]=new_session(session_name)
    return remove(image,
                alpha_matting, 
                alpha_matting_foreground_threshold, 
                alpha_matting_background_threshold, 
                alpha_matting_erode_size, 
                sessions[session_name], 
                only_mask,
                post_process_mask)
class Script(scripts.Script):
    def title(self):
        return "Auto Mask"
    def show(self, is_img2img):
        return is_img2img
    def ui(self, is_img2img):
        if not is_img2img: return
        alpha_matting=components.Checkbox(label="Alpha Matting")
        alpha_matting_foreground_threshold=components.Slider(minimum=0, maximum=255,step=1, default=240, label="Alpha Matting Foreground Threshold")
        alpha_matting_background_threshold=components.Slider(minimum=0, maximum=255,step=1, default=10, label="Alpha Matting Background Threshold")
        alpha_matting_erode_size=components.Slider(minimum=0, maximum=255,step=1, default=10, label="Alpha Matting Erode Size")
        session_name=components.Dropdown(["u2net", "u2netp","u2net_human_seg","u2net_cloth_seg","silueta"], label="Session")
        only_mask=components.Checkbox(label="Only Mask")
        post_process_mask=components.Checkbox(label="Post Process Mask")
        with Blocks() as demo:
            with Row(equal_height=True):
                image=components.Image(type="pil")
                mask=components.Image(type="pil")
        btn = components.Button(label="Preview Remove Background")
        if image is not None:
            btn.click(remove_background, inputs=[image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,\
                                                 alpha_matting_erode_size,session_name,only_mask,post_process_mask], outputs=[mask])
        return [image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,alpha_matting_erode_size,session_name,\
                only_mask,post_process_mask]
        # alpha_matting=gr.inputs.Checkbox(label="Alpha Matting")
        # alpha_matting_foreground_threshold=gr.inputs.Slider(minimum=0, maximum=255,step=1, default=240, label="Alpha Matting Foreground Threshold")
        # alpha_matting_background_threshold=gr.inputs.Slider(minimum=0, maximum=255,step=1, default=10, label="Alpha Matting Background Threshold")
        # alpha_matting_erode_size=gr.inputs.Slider(minimum=0, maximum=255,step=1, default=10, label="Alpha Matting Erode Size")
        # session_name=gr.inputs.Dropdown(["u2net", "u2netp","u2net_human_seg","u2net_cloth_seg","silueta"], label="Session")
        # only_mask=gr.inputs.Checkbox(label="Only Mask")
        # post_process_mask=gr.inputs.Checkbox(label="Post Process Mask")
        # with gr.Blocks() as demo:
        #     with gr.Row().style(equal_height=True):
        #         image=gr.Image(type="pil")
        #         mask=gr.Image(type="pil")
        # btn = gr.Button(value="Preview Remove Background")
        # if image is not None:
        #     btn.click(remove_background, inputs=[image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,\
        #                                          alpha_matting_erode_size,session_name,only_mask,post_process_mask], outputs=[mask])
        # return [image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,alpha_matting_erode_size,session_name,\
        #         only_mask,post_process_mask]
    def run(self,p,image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,alpha_matting_erode_size,session_name,\
                only_mask,post_process_mask):
        if image is None:
            image=p.init_images[0]
        only_mask=True
        mask=remove_background(image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,\
                                                 alpha_matting_erode_size,session_name,only_mask,post_process_mask)
        p.image_mask=mask
        proc = process_images(p)
        proc.images.append(mask)
        return proc

def auto_mask_api(_: Blocks, app: FastAPI):
    @app.get('/figma/healthcheck', status_code=status.HTTP_200_OK)
    def perform_healthcheck():
        return {'healthcheck': 'Everything OK!'}
    @app.get("/figma/status", status_code=status.HTTP_200_OK)
    async def get_status():
        return {"status": "ok", "version": "1.0.0"}
    @app.post("/figma/auto_mask/remove-background")
    async def post_remove_background(image_str: str = Body(...), alpha_matting: bool = Body(...), alpha_matting_foreground_threshold: int = Body(...),\
                                alpha_matting_background_threshold: int = Body(...), alpha_matting_erode_size: int = Body(...), session_name: str = Body(...),\
                                only_mask: bool = Body(...), post_process_mask: bool = Body(...)):
        import base64
        import io
        image_bytes = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_bytes),formats=["PNG"])
        mask=remove_background(image,alpha_matting,alpha_matting_foreground_threshold,alpha_matting_background_threshold,\
                                                 alpha_matting_erode_size,session_name,only_mask,post_process_mask)
        buffered = io.BytesIO()
        mask.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        return {"mask": img_str}
try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(auto_mask_api)
except:
    pass