## 风格转换

### 图像风格转换 OR 视频风格转换

### 部署服务

```shell
git clone 本项目地址
pip install -r requirements.txt
uvicorn main:app --reload
```

### Docker Package

```shell
docker build -t styleconvert:latest .
docker run -it -p 9002:9002 styleconvert:latest
```