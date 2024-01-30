import argparse
import os
import cv2
from PIL import Image
from tqdm import tqdm
import numpy as np
import queue
import threading
import onnxruntime as ort

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class Videocap:
    def __init__(self, video, model_name, limit=1280):
        self.model_name = model_name
        vid = cv2.VideoCapture(video)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = vid.get(cv2.CAP_PROP_FPS)
        self.ori_width, self.ori_height = width, height

        max_edge = max(width, height)
        # Prevent GPU memory from overflowing due to excessive input size.
        scale_factor = limit / max_edge if max_edge > limit else 1.
        height = int(round(height * scale_factor))
        width = int(round(width * scale_factor))
        self.width, self.height = self.to_8s(width), self.to_8s(height)

        self.count = 0  # Records the number of frames entered into the queue.
        self.cap = vid
        self.ret, frame = self.cap.read()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.q = queue.Queue(maxsize=100)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            # print("get me")
            self.ret, frame = self.cap.read()
            if not self.ret:
                break
            frame = np.asarray(self.process_frame(frame, self.width, self.height))
            self.q.put(frame)
            self.count += 1
        self.cap.release()

    def read(self):
        f = self.q.get()
        self.q.task_done()
        return f

    def to_8s(self, x):
        if 'tiny' in self.model_name:
            return 256 if x < 256 else x - x % 16
        else:
            return 256 if x < 256 else x - x % 8

    def process_frame(self, img, width, height):
        img = Image.fromarray(img[:, :, ::-1]).resize((width, height), Image.ANTIALIAS)
        img = np.array(img).astype(np.float32) / 127.5 - 1.0
        return np.expand_dims(img, axis=0)


class Cartoonizer:
    def __init__(self, input_video_path, model_path, output, device):
        # print("arg:", arg)
        # self.args = arg

        if ort.get_device() == 'GPU' and device == "gpu":
            self.sess_land = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider', ])
        else:
            self.sess_land = ort.InferenceSession(model_path, providers=['CPUExecutionProvider', ])
        self.name = os.path.basename(model_path).rsplit('.', 1)[0]

    def post_precess(self, img, wh):
        img = (img.squeeze() + 1.) / 2 * 255
        img = img.clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img).resize((wh[0], wh[1]), Image.ANTIALIAS)
        img = np.array(img).astype(np.uint8)
        return img

    def __call__(self):
        # load video
        vid = Videocap(input_video_path, self.name)
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        num = vid.total
        self.video_out = cv2.VideoWriter(os.path.join(output, os.path.basename(input_video_path).rsplit('.', 1)[0] + f'_{self.name}.mp4'), codec, vid.fps,
                                         (vid.ori_width, vid.ori_height))
        pbar = tqdm(total=vid.total, )
        pbar.set_description(
            f"Running: {os.path.basename(input_video_path).rsplit('.', 1)[0] + f'_{self.name}.mp4'}")
        while num > 0:
            if vid.count < vid.total and vid.ret == False and vid.q.empty():
                pbar.close()
                self.video_out.release()
                return "The video is broken, please upload the video again."
            frame = vid.read()
            fake_img = self.sess_land.run(None, {self.sess_land.get_inputs()[0].name: frame})[0]
            fake_img = self.post_precess(fake_img, (vid.ori_width, vid.ori_height))
            self.video_out.write(fake_img[:, :, ::-1])
            pbar.update(1)
            num -= 1
        pbar.close()
        self.video_out.release()
        return os.path.join(output, os.path.basename(input_video_path).rsplit('.', 1)[0] + f'_{self.name}.mp4')


if __name__ == '__main__':
    input_video_path = './assets/video/' + '1.mp4'
    model_path = 'models/AnimeGANv3_Hayao_36.onnx'
    output = './assets/output/'
    func = Cartoonizer(input_video_path, model_path, output, device='cpu')
    info = func()
    print(f'output video: {info}')
