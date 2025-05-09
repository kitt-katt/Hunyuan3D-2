import asyncio
import pynvml
from PIL import Image
import time
import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.responses import FileResponse
from typing import Dict, List
import shutil
from collections import deque
import torch
import trimesh

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline


app = FastAPI()
pipeline_turbo_mesh = None
pipeline_multiview = None
pipeline_turbo_tex = None

# Directories to store results and images
RES_DIR = 'res'
IMAGES_DIR = 'images'
MESH_DIR = 'meshes'
os.makedirs(RES_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MESH_DIR, exist_ok=True)

# Queue for processing requests (both mesh generation and texturing)
processing_queue: deque = deque()
current_task_id: str = None
is_processing: bool = False
# Dictionary to store errors for tasks
error_tasks: Dict[str, str] = {}
# Dictionary to store task type for accurate status reporting
task_types: Dict[str, str] = {}

async def get_gpu_usage():
    """
    Возвращает загруженность GPU в процентах.
    Если GPU недоступен или произошла ошибка, возвращает 0.
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        return 0
    except Exception as e:
        print(f"Ошибка при получении загруженности GPU: {e}")
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


async def get_gpu_memory_usage():
    """
    Возвращает использование видеопамяти GPU в процентах.
    Если GPU недоступен или произошла ошибка, возвращает 0.
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return (mem_info.used / mem_info.total) * 100
        return 0
    except Exception as e:
        print(f"Ошибка при получении использования памяти GPU: {e}")
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def texture_for_mesh(images_path: List[str], mesh_path: str) -> str:
    images = []
    for image_path in images_path:
        image = Image.open(image_path)
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)
        images.append(image)
    
    mesh = trimesh.load(mesh_path)
    
    mesh = pipeline_turbo_tex(mesh, image=images)
    task_id = os.path.basename(mesh_path).split('.')[0].split('_')[0]
    output_path = os.path.join(RES_DIR, f"{task_id}.glb")
    mesh.export(output_path)
    return output_path


def single_image_mesh_generation(image_path: str) -> str:
    image = Image.open(image_path).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    mesh = pipeline_turbo_mesh(
        image=image, 
        num_inference_steps=50,
        octree_resolution=380,
        num_chunks=200000,
        generator=torch.manual_seed(12345),
        output_type='trimesh'
    )[0]
    task_id = os.path.basename(image_path).split('.')[0].split('_')[0]
    output_path = os.path.join(RES_DIR, f"{task_id}.glb")
    mesh.export(output_path)
    return output_path


def multi_image_mesh_generation(images: Dict[str, str]) -> str:
    
    task_id = os.path.basename(list(images.values())[0]).split('.')[0].split('_')[0]
    
    for key in images:
        image = Image.open(images[key]).convert("RGBA")
        if image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)
        images[key] = image
    
    # Use the first image as primary for mesh generation if needed
    mesh = pipeline_multiview(
        image=images, 
        num_inference_steps=35,
        octree_resolution=380,
        num_chunks=200000,
        generator=torch.manual_seed(12345),
        output_type='trimesh'
    )[0]
    
    output_path = os.path.join(RES_DIR, f"{task_id}.glb")
    mesh.export(output_path)
    return output_path


def load_pipeline():
    model_path = 'tencent/Hunyuan3D-2'
    pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder='Hunyuan3D-DiT-v2-0-Turbo',
        use_safetensors=True
    )
    pipeline_multiview = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mv',
        subfolder='hunyuan3d-dit-v2-mv-turbo',
        variant='fp16'
    )
    pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(
        model_path,
        subfolder='hunyuan3d-paint-v2-0'
    )
    pipeline_shapegen.enable_flashvdm()
    pipeline_multiview.enable_flashvdm()
    return pipeline_shapegen, pipeline_multiview, pipeline_texgen


async def process_queue():
    global current_task_id, is_processing
    while True:
        if processing_queue and not is_processing:
            is_processing = True
            task_id, task_data, task_type = processing_queue.popleft()
            current_task_id = task_id
            task_types[task_id] = task_type
            print(f"Обработка задачи {task_id} (Тип: {task_type})")
            try:
                if task_type == "mesh_generation":
                    await asyncio.to_thread(single_image_mesh_generation, task_data['image_path'])
                elif task_type == "texturing":
                    await asyncio.to_thread(texture_for_mesh, task_data['images_path'], task_data['mesh_path'])
                elif task_type == "multi_mesh_generation":
                    await asyncio.to_thread(multi_image_mesh_generation, task_data['images_path'])
            except Exception as e:
                print(f"Ошибка при обработке задачи {task_id}: {e}")
                error_tasks[task_id] = str(e)
            finally:
                current_task_id = None
                is_processing = False
                if task_id in task_types:
                    del task_types[task_id]
        else:
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    global pipeline_turbo_mesh, pipeline_turbo_tex, pipeline_multiview
    pipeline_turbo_mesh, pipeline_multiview, pipeline_turbo_tex = load_pipeline()
    asyncio.create_task(process_queue())
    print("Сервер запущен, пайплайны загружены, обработка очереди начата.")


@app.post("/single_to_mesh/")
async def single_to_mesh(file: UploadFile = File(...)):
    task_id = str(uuid.uuid4())
    image_path = os.path.join(IMAGES_DIR, f"{task_id}_{file.filename}")
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    processing_queue.append((task_id, {'image_path': image_path}, "mesh_generation"))
    return {"task_id": task_id, "message": "Image uploaded and queued for processing"}


@app.post("/multi_to_mesh/")
async def multi_to_mesh(
    front: UploadFile = File(None),
    back: UploadFile = File(None),
    left: UploadFile = File(None),
    right: UploadFile = File(None)
):
    task_id = str(uuid.uuid4())
    images_path = {}
    
    # Map the uploaded files to their respective keys
    uploaded_files = {
        "front": front,
        "back": back,
        "left": left,
        "right": right
    }
    
    # Validate and store paths for provided images
    for key, image_file in uploaded_files.items():
        if image_file is not None:
            image_path = os.path.join(IMAGES_DIR, f"{task_id}_{key}_{image_file.filename}")
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)
            images_path[key] = image_path
    
    # Check if at least one image is provided
    if len(images_path) < 1:
        raise HTTPException(status_code=400, detail="At least one image (front, back, left, or right) is required")
        
    from pprint import pprint
    pprint(images_path)
    processing_queue.append((task_id, {'images_path': images_path}, "multi_mesh_generation"))
    return {"task_id": task_id, "message": "Images uploaded and queued for mesh generation"}


@app.post("/texture_mesh/")
async def texture_mesh(mesh_file: UploadFile = File(...), image_files: List[UploadFile] = File(...)):
    task_id = str(uuid.uuid4())
    mesh_path = os.path.join(MESH_DIR, f"{task_id}_{mesh_file.filename}")
    with open(mesh_path, "wb") as buffer:
        shutil.copyfileobj(mesh_file.file, buffer)
    
    images_path = []
    for image_file in image_files:
        image_path = os.path.join(IMAGES_DIR, f"{task_id}_{image_file.filename}")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        images_path.append(image_path)
    
    processing_queue.append((task_id, {'mesh_path': mesh_path, 'images_path': images_path}, "texturing"))
    return {"task_id": task_id, "message": "Mesh and images uploaded and queued for texturing"}


@app.get("/get_mesh/{task_id}")
async def get_mesh(task_id: str):
    mesh_path = os.path.join(RES_DIR, f"{task_id}.glb")
    if os.path.exists(mesh_path):
        return FileResponse(mesh_path, media_type="application/octet-stream", filename=f"{task_id}.glb")
    else:
        if task_id == current_task_id:
            return {"status": "Processing", "message": "Your model is currently being processed."}
        elif any(t[0] == task_id for t in processing_queue):
            return {"status": "Queued", "message": "Your model is in the queue for processing."}
        elif task_id in error_tasks:
            return {"status": "Error", "message": f"An error occurred during processing: {error_tasks[task_id]}"}
        else:
            raise HTTPException(status_code=404, detail="Model not found or task ID invalid.")


@app.get("/status/{task_id}")
async def get_status(task_id: str):
    mesh_path = os.path.join(RES_DIR, f"{task_id}.glb")
    if os.path.exists(mesh_path):
        return {"status": "Completed", "message": "Model is ready for download."}
    elif task_id in error_tasks:
        return {"status": "Error", "message": f"An error occurred during processing: {error_tasks[task_id]}"}
    elif task_id == current_task_id:
        task_type = task_types.get(task_id, "unknown")
        message = "Model is currently being processed." if task_type == "mesh_generation" else "Mesh is currently being textured."
        return {"status": "Processing", "message": message}
    elif any(t[0] == task_id for t in processing_queue):
        return {"status": "Queued", "message": "Task is in the queue for processing."}
    else:
        raise HTTPException(status_code=404, detail="Task not found or ID invalid.")


@app.delete("/delete_mesh/{task_id}")
async def delete_mesh(task_id: str):
    mesh_path = os.path.join(RES_DIR, f"{task_id}.glb")
    if os.path.exists(mesh_path):
        os.remove(mesh_path)
        return {"message": f"Model {task_id} deleted successfully."}
    else:
        raise HTTPException(status_code=404, detail="Model not found.")
    

@app.get("/get_gpu_usage")
async def get_gpu_usage_endpoint():
    return await get_gpu_usage()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
