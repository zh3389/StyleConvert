import os.path
import shutil
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, Request, File, UploadFile
from convertImages import ConvertStyle
from convertVideo import ConvertVideo

app = FastAPI()
convertStyle = ConvertStyle()

input_imgs_path = './assets/test'
output_path = './assets/output/'


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """捕获默认框架返回的异常信息格式，修改为规定格式"""
    return JSONResponse({"success": False,
                         "code": 400,
                         "msg": f"接口参数传递错误",
                         "data": exc.errors()})


@app.post("/convert/v1/convertImage")
async def upload_image(imgFile: UploadFile = File(...), model_name: str = 'hayao', device: str = 'cpu'):
    """
    转换图像的风格 动漫风：hayao  日本脸风：jp_face  素描：sketch  新海风：shinkai  可爱风：cute
    :param imgFile: 输入一张图像
    :param model_name: 风格模型的名称 【AnimeGANv3_Hayao_36, AnimeGANv3_JP_face_v1.0, AnimeGANv3_PortraitSketch, AnimeGANv3_Shinkai_37, AnimeGANv3_tiny_Cute】
    :param device: cpu or gpu
    :return: 图像的下载链接
    """
    model_dict = {"hayao": "AnimeGANv3_Hayao_36.onnx",
                  "jp_face": "AnimeGANv3_JP_face_v1.0.onnx",
                  "sketch": "AnimeGANv3_PortraitSketch.onnx",
                  "shinkai": "AnimeGANv3_Shinkai_37.onnx",
                  "cute": "AnimeGANv3_tiny_Cute.onnx"}
    onnx_file = f'models/{model_dict[model_name]}'
    try:
        # 保存上传的文件到临时目录
        img_path = os.path.join("/tmp", imgFile.filename)
        output_dir = "/tmp/output"
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(imgFile.file, buffer)
        download_img_link = convertStyle.ConvertImage(img_path, output_dir, onnx_file, device=device)

        # 返回处理后的图像的下载链接
        return FileResponse(download_img_link, filename=imgFile.filename, media_type="application/octet-stream")
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


@app.post("/convert/v1/convertVideo")
async def upload_video(videoFile: UploadFile = File(...), model_name: str = 'hayao', device: str = 'cpu'):
    """
    转换视频的风格 动漫风：hayao  日本脸风：jp_face  素描：sketch  新海风：shinkai  可爱风：cute
    :param videoFile: 输入一段视频
    :param model_name: 风格模型的名称 【AnimeGANv3_Hayao_36, AnimeGANv3_JP_face_v1.0, AnimeGANv3_PortraitSketch, AnimeGANv3_Shinkai_37, AnimeGANv3_tiny_Cute】
    :param device: cpu or gpu
    :return: 视频的下载链接
    """
    model_dict = {"hayao": "AnimeGANv3_Hayao_36.onnx",
                  "jp_face": "AnimeGANv3_JP_face_v1.0.onnx",
                  "sketch": "AnimeGANv3_PortraitSketch.onnx",
                  "shinkai": "AnimeGANv3_Shinkai_37.onnx",
                  "cute": "AnimeGANv3_tiny_Cute.onnx"}
    onnx_file = f'models/{model_dict[model_name]}'
    try:
        # 保存上传的文件到临时目录
        video_path = os.path.join("/tmp", videoFile.filename)
        output_dir = "/tmp/output"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(videoFile.file, buffer)

        convertVideo = ConvertVideo(video_path, output_dir, onnx_file, device=device)
        download_video_link = convertVideo()
        # 返回处理后的视频的下载链接
        return FileResponse(download_video_link, filename=videoFile.filename, media_type="application/octet-stream")
    except Exception as e:
        return {"success": False,
                "code": 500,
                "msg": str(e),
                "data": None}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9002)
