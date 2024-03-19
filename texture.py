import numpy as np
import cv2
import torch
import PIL
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
# import os
# import contextlib
# from io import StringIO
from tqdm.auto import tqdm
import signal
import requests
import urllib.request
import urllib.parse
import os
import re

# from realesrgan import RealESRGANer
# from basicsr.archs.rrdbnet_arch import RRDBNet

def download_file(
      link: str,
      path: str,
      block_size: int = 1024,
      force_download: bool = False,
      progress: bool = True,
      interrupt_check: bool = True
) -> str:

  def truncate_string(string: str, length: int):
    length -= 5 if length - 5 > 0 else 0
    curr_len = len(string)
    new_len = len(string[:length // 2] + "(...)" + string[-length // 2:])
    if new_len > curr_len:
      return string
    else:
      return string[:length // 2] + "(...)" + string[-length // 2:]

  def remove_char(string: str, chars: list):
    for char in chars:
      string = string.replace(char, "")
    return string

  # source: https://github.com/wkentaro/gdown/blob/main/gdown/download.py
  def google_drive_parse_url(url: str):
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    is_gdrive = parsed.hostname in ["drive.google.com", "docs.google.com"]
    is_download_link = parsed.path.endswith("/uc")

    if not is_gdrive:
      return is_gdrive, is_download_link

    file_id = None
    if "id" in query:
      file_ids = query["id"]
      if len(file_ids) == 1:
        file_id = file_ids[0]
    else:
      patterns = [r"^/file/d/(.*?)/view$", r"^/presentation/d/(.*?)/edit$"]
      for pattern in patterns:
        match = re.match(pattern, parsed.path)
        if match:
          file_id = match.groups()[0]
          break

    return file_id, is_download_link

  # source: https://github.com/wkentaro/gdown/blob/main/gdown/download.py
  def get_url_from_gdrive_confirmation(contents: str):
    url = ""
    for line in contents.splitlines():
      m = re.search(r'href="(/uc\?export=download[^"]+)', line)
      if m:
        url = "https://docs.google.com" + m.groups()[0]
        url = url.replace("&amp;", "&")
        break
      m = re.search('id="download-form" action="(.+?)"', line)
      if m:
        url = m.groups()[0]
        url = url.replace("&amp;", "&")
        break
      m = re.search('"downloadUrl":"([^"]+)', line)
      if m:
        url = m.groups()[0]
        url = url.replace("\\u003d", "=")
        url = url.replace("\\u0026", "&")
        break
      m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
      if m:
        error = m.groups()[0]
        raise RuntimeError(error)
    if not url:
      raise RuntimeError(
        "Cannot retrieve the link of the file. "
      )
    return url

  def interrupt(*args):
    if os.path.isfile(filepath):
      os.remove(filepath)
    raise KeyboardInterrupt

  # create folder if not exists
  if not os.path.exists(path):
    os.makedirs(path)

  # check if link is google drive link
  if not google_drive_parse_url(link)[0]:
    response = requests.get(link, stream=True, allow_redirects=True)
  else:
    if not google_drive_parse_url(link)[1]:
      # convert to direct link
      file_id = google_drive_parse_url(link)[0]
      link = f"https://drive.google.com/uc?id={file_id}"
    # test if redirect is needed
    response = requests.get(link, stream=True, allow_redirects=True)
    if response.headers.get("Content-Disposition") is None:
      page = urllib.request.urlopen(link)
      link = get_url_from_gdrive_confirmation(str(page.read()))
      response = requests.get(link, stream=True, allow_redirects=True)

  if response.status_code == 404:
    raise FileNotFoundError(f"File not found at {link}")

  # get filename
  content_disposition = response.headers.get("Content-Disposition")
  if content_disposition:
    filename = re.findall(r'filename=(.*?)(?:[;\n]|$)', content_disposition)[0]
  else:
    filename = os.path.basename(link)

  filename = remove_char(filename, ['/', '\\', ':', '*', '?', '"', "'", '<', '>', '|', ';'])
  filename = filename.replace(' ', '_')

  filepath = os.path.join(path, filename)

  # download file
  if os.path.isfile(filepath) and not force_download:
    print(f"{filename} already exists. Skipping download.")
  else:
    text = f"Downloading {truncate_string(filename, 50)}"
    with open(filepath, "wb") as file:
      total_size = int(response.headers.get("content-length", 0))
      with tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=text,
        unit_divisor=1024,
        disable=not progress,
      ) as pb:
        if interrupt_check:
          signal.signal(signal.SIGINT, lambda signum, frame: interrupt())
        for data in response.iter_content(block_size):
          pb.update(len(data))
          file.write(data)
  del response
  return filename
def factorize(num: int, max_value: int) -> list[float]:
  result = []
  while num > max_value:
    result.append(max_value)
    num /= max_value
  result.append(round(num, 4))
  return result

#
# def upscale(
#     imgs: list[PIL.Image.Image],
#     model_name: str = "RealESRGAN_x4plus",
#     scale_factor: float = 4,
#     half_precision: bool = False,
#     tile: int = 0,
#     tile_pad: int = 10,
#     pre_pad: int = 0,
# ) -> list[PIL.Image.Image]:
#
#   # check model
#   if model_name == "RealESRGAN_x4plus":
#     upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
#     netscale = 4
#     file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
#   elif model_name == "RealESRNet_x4plus":
#     upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
#     netscale = 4
#     file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
#   elif model_name == "RealESRGAN_x4plus_anime_6B":
#     upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
#     netscale = 4
#     file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
#   elif model_name == "RealESRGAN_x2plus":
#     upscale_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
#     netscale = 2
#     file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
#   else:
#     raise NotImplementedError("Model name not supported")
#
#   # download model
#   model_path = download_file(file_url, path="./upscaler-model", progress=False, interrupt_check=False)
#
#   # declare the upscaler
#   upsampler = RealESRGANer(
#     scale=netscale,
#     model_path=os.path.join("./upscaler-model", model_path),
#     dni_weight=None,
#     model=upscale_model,
#     tile=tile,
#     tile_pad=tile_pad,
#     pre_pad=pre_pad,
#     half=half_precision,
#     gpu_id=None
#   )
#
#   # upscale
#   torch.cuda.empty_cache()
#   upscaled_imgs = []
#   with tqdm(total=len(imgs)) as pb:
#     for i, img in enumerate(imgs):
#       img = np.array(img)
#       outscale_list = factorize(scale_factor, netscale)
#       with contextlib.redirect_stdout(StringIO()):
#         for outscale in outscale_list:
#           curr_img = upsampler.enhance(img, outscale=outscale)[0]
#           img = curr_img
#         upscaled_imgs.append(PIL.Image.fromarray(img))
#
#       pb.update(1)
#   torch.cuda.empty_cache()
#
#   return upscaled_imgs



def is_inside(inner, outer):
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

def is_small(bbox, thre=50):
    if bbox[2]-bbox[0]<thre or bbox[3]-bbox[1]<thre:
        return True
    else:
        return False


depth2img_pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
).to("cuda")

in_paint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            revision="fp16",
            torch_dtype=torch.float16,
        )
in_paint_pipe = in_paint_pipe.to("cuda")
in_paint_pipe.safety_checker = lambda images, clip_input: (images, False)

import random

def load_kitchen_texture(image, test_name, object_id, bboxes):
    # load the image
    # get the bounding box for drawer and doors
    texture_path = "textures/{0}".format(test_name)
    os.makedirs(texture_path + '/{0}'.format(object_id), exist_ok=True)

    # create folder for the texture
    side_texture = "default_textures/inside.jpg"
    side_image = cv2.imread(side_texture)
    texture_list = []
    for bbox_id, each_bbox in enumerate(bboxes):
        if os.path.exists(texture_path + "/{0}/{1}.png".format(object_id,  bbox_id)):
            texture_list.append(texture_path + "/{0}/{1}.png".format(object_id, bbox_id))
            continue
        threshold1 = 0
        front_image = image[each_bbox[0]+threshold1:each_bbox[2]-threshold1, each_bbox[1]+threshold1:each_bbox[3]-threshold1]
        w,h,_ = front_image.shape

        inside_bbox = []
        for inside_id, each_bbox1 in enumerate(bboxes):
            if inside_id ==bbox_id:
                continue
            if is_inside(each_bbox1, each_bbox):
                inside_bbox.append(each_bbox1)

        # resize everything to 512x512
        inpaint_img = PIL.Image.fromarray(front_image).resize((512, 512))
        inpaint_mask = np.zeros((w,h))
        threshold = 1
        for each_inside_bbox in inside_bbox:
            inpaint_mask[max(0, each_inside_bbox[0] - each_bbox[0]-threshold):each_inside_bbox[2] - each_bbox[0]+threshold, max(0, each_inside_bbox[1] - each_bbox[1]-threshold):each_inside_bbox[3] - each_bbox[1]+threshold]=255

        inpaint_mask = PIL.Image.fromarray(cv2.resize(inpaint_mask, (512, 512), interpolation=cv2.INTER_NEAREST))

        # impaint the texture
        # inpaint_img = upscale([inpaint_img])[0].resize((512, 512))
        new_image = in_paint_pipe(prompt="panel texture, original color, smooth texture, Intricately Detailed, 16k, natural lighting, Best Quality, Masterpiece, photorealistic", image=inpaint_img, mask_image=inpaint_mask).images[0]

        if not is_small(each_bbox, 10): # use the drawer color to be the base
            base_image = np.array(new_image.resize((200, 200)))
            base_texture = np.zeros((600, 600, 3))
            base_texture[200:400, :200, :] = np.rot90(base_image)
            base_texture[400:600, 400:600, :] = np.rot90(base_image)
            base_texture[200:400, 200:400, :] = np.rot90(base_image)
            base_texture[400:600, 200:400, :] = np.rot90(base_image)
            base_texture[200:400, 400:600, :] = np.rot90(base_image)
            base_texture[400:600, :200, :] = np.rot90(base_image)
            PIL.Image.fromarray(base_texture.astype(np.uint8)).save(
                texture_path + "/{0}/base.png".format(object_id))

        new_image = np.array(new_image.resize((200, 200)))
        # putting this together with side images.
        texture_map = np.zeros((600, 600, 3))
        texture_map[200:400, :200, :] = np.rot90(new_image)
        texture_map[400:600, 400:600, :] = np.rot90(new_image)
        texture_map[200:400, 200:400, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        texture_map[400:600, 200:400, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        texture_map[200:400, 400:600, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        texture_map[400:600, :200, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        # save
        PIL.Image.fromarray(texture_map.astype(np.uint8)).save(
            texture_path + "/{0}/{1}.png".format(object_id, bbox_id))
        texture_list.append(texture_path + "/{0}/{1}.png".format(object_id, bbox_id))
    return texture_list


import glob


def load_texture(img_path, label_path, if_random=False):
    pred_path = label_path
    image = np.array(PIL.Image.open(img_path).convert("RGB"))
    # get the bounding box for drawer and doors
    object_info = np.load(pred_path, allow_pickle=True).item()
    bboxes = object_info['part_normalized_bbox']
    test_name = os.path.basename(img_path)[:-4]
    texture_path = "textures"
    os.makedirs(texture_path+'/{0}'.format(test_name), exist_ok=True)

    # create folder for the texture
    side_texture = "default_textures/inside.jpg"
    side_image = cv2.imread(side_texture)
    texture_list = []


    has_drawer = False
    for bbox_id, each_bbox in enumerate(bboxes):
        each_bbox = [int(each_bbox[0] * image.shape[0]),
                      int(each_bbox[1] * image.shape[1]),
                      int((each_bbox[0] + each_bbox[2]) * image.shape[0]),
                      int((each_bbox[1] + each_bbox[3]) * image.shape[1]),
                      ]

        front_image = image[each_bbox[0]:each_bbox[2],
                      each_bbox[1]:each_bbox[3]]
        w, h, _ = front_image.shape
        if w < h:
            if not is_small(each_bbox):
                has_drawer=True
                break

    for bbox_id, each_bbox in enumerate(bboxes):
        if len(glob.glob(texture_path + f"/{test_name}/*")) ==len(bboxes)+1:
            texture_list.append(texture_path + f"/{test_name}/{bbox_id}.png")
            continue
        each_bbox = [int(each_bbox[0] * image.shape[0]),
                     int(each_bbox[1] * image.shape[1]),
                     int((each_bbox[0] + each_bbox[2]) * image.shape[0]),
                     int((each_bbox[1] + each_bbox[3]) * image.shape[1]),
                     ]



        # get the front image
        min_th = min(each_bbox[2] - each_bbox[0], each_bbox[3] - each_bbox[1])

        if min_th<25:
            front_image = image[each_bbox[0]:each_bbox[2],
                          each_bbox[1]:each_bbox[3]]
        else:
            if is_small(each_bbox):
                threshold1 = int(min(min_th/2, 5))
            else:
                threshold1 = int(min(min_th/2, 10))

            front_image = image[each_bbox[0]+threshold1:each_bbox[2]-threshold1, each_bbox[1]+threshold1:each_bbox[3]-threshold1]

        # remove handle
        # get any bbox inside this bbox
        w,h,_ = front_image.shape

        inside_bbox = []
        for inside_id, each_bbox1 in enumerate(bboxes):
            if inside_id ==bbox_id:
                continue
            if is_inside(each_bbox1, each_bbox):
                inside_bbox.append(each_bbox1)

        # resize everything to 512x512
        inpaint_img = PIL.Image.fromarray(front_image).resize((512, 512))
        inpaint_mask = np.zeros((w,h))
        threshold = 25
        for each_inside_bbox in inside_bbox:
            inpaint_mask[max(0, each_inside_bbox[0] - each_bbox[0]-threshold):each_inside_bbox[2] - each_bbox[0]+threshold, max(0, each_inside_bbox[1] - each_bbox[1]-threshold):each_inside_bbox[3] - each_bbox[1]+threshold]=255

        inpaint_mask = PIL.Image.fromarray(cv2.resize(inpaint_mask, (512, 512), interpolation=cv2.INTER_NEAREST))

        # impaint the texture
        # inpaint_img = upscale([inpaint_img])[0].resize((512, 512))
        text_promt = random.choice(['bright light', 'natural light', 'ultra smooth', 'good quality wood', 'nice pattern', 'wooden pattern'])
        new_image = in_paint_pipe(prompt="just pure flat wood panel, smooth texture, Intricately Detailed, 16k, natural lighting, Best Quality, Masterpiece, photorealistic", image=inpaint_img, mask_image=inpaint_mask).images[0]
        if if_random:
            n_propmt = "bad, deformed, ugly, bad anotomy, low resolution"
            new_image = depth2img_pipe(prompt=text_promt+" wood pattern", image=new_image, negative_prompt=n_propmt, strength=0.8).images[0]

        if has_drawer:
            if w<h and not is_small(each_bbox): # use the drawer color to be the base
                base_image = np.array(new_image.resize((200, 200)))
                base_texture = np.zeros((600, 600, 3))
                base_texture[200:400, :200, :] = np.rot90(base_image)
                base_texture[400:600, 400:600, :] = np.rot90(base_image)
                base_texture[200:400, 200:400, :] = np.rot90(base_image)
                base_texture[400:600, 200:400, :] = np.rot90(base_image)
                base_texture[200:400, 400:600, :] = np.rot90(base_image)
                base_texture[400:600, :200, :] = np.rot90(base_image)
                PIL.Image.fromarray(base_texture.astype(np.uint8)).save(texture_path + f"/{test_name}/base.png")

        else:
            if w>h and not is_small(each_bbox): # use the drawer color to be the base
                base_image = np.array(new_image.resize((200, 200)))
                base_texture = np.zeros((600, 600, 3))
                base_texture[200:400, :200, :] = np.rot90(base_image)
                base_texture[400:600, 400:600, :] = np.rot90(base_image)
                base_texture[200:400, 200:400, :] = np.rot90(base_image)
                base_texture[400:600, 200:400, :] = np.rot90(base_image)
                base_texture[200:400, 400:600, :] = np.rot90(base_image)
                base_texture[400:600, :200, :] = np.rot90(base_image)
                PIL.Image.fromarray(base_texture.astype(np.uint8)).save(texture_path + f"/{test_name}/base.png")

        new_image = np.array(new_image.resize((200, 200)))
        # putting this together with side images.
        # create a texture map 512 by 512
        texture_map = np.zeros((600, 600, 3))


        texture_map[200:400, :200, :] = np.rot90(new_image)
        texture_map[400:600, 400:600, :] = np.rot90(new_image)
        texture_map[200:400, 200:400, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        texture_map[400:600, 200:400, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        texture_map[200:400, 400:600, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        texture_map[400:600, :200, :] = np.array(PIL.Image.fromarray(side_image).resize((200, 200)))
        # save
        PIL.Image.fromarray(texture_map.astype(np.uint8)).save(texture_path+f"/{test_name}/{bbox_id}.png")
    return texture_list
