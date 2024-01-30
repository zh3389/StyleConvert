# -*- coding: utf-8 -*-
import onnxruntime as ort
import time, os, cv2
import numpy as np
from glob import glob


class ConvertStyle:
    def __init__(self):
        self.model_list = ['AnimeGANv3_tiny_Cute.onnx', 'AnimeGANv3_Hayao_36.onnx', 'AnimeGANv3_JP_face_v1.0.onnx',
                           'AnimeGANv3_PortraitSketch.onnx', 'AnimeGANv3_Shinkai_37.onnx']
        self.pic_form = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']

    @staticmethod
    def check_folder(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def process_image(self, img, model_name):
        h, w = img.shape[:2]

        # resize image to multiple of 8s
        def to_8s(x):
            # If using the tiny model, the multiple should be 16 instead of 8.
            if 'tiny' in os.path.basename(model_name):
                return 256 if x < 256 else x - x % 16
            else:
                return 256 if x < 256 else x - x % 8

        img = cv2.resize(img, (to_8s(w), to_8s(h)))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return img

    def load_test_data(self, image_path, model_name):
        img0 = cv2.imread(image_path).astype(np.float32)
        img = self.process_image(img0, model_name)
        img = np.expand_dims(img, axis=0)
        return img, img0.shape

    def save_images(self, images, image_path, size):
        images = (np.squeeze(images) + 1.) / 2 * 255
        images = np.clip(images, 0, 255).astype(np.uint8)
        images = cv2.resize(images, size)
        cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))

    def Convert(self, input_imgs_path, output_path, onnx="model.onnx", device="cpu"):
        # result_dir = opj(output_path, style_name)
        result_dir = output_path
        self.check_folder(result_dir)
        test_files = glob('{}/*.*'.format(input_imgs_path))
        test_files = [x for x in test_files if os.path.splitext(x)[-1] in self.pic_form]
        if ort.get_device() == 'GPU' and device == "gpu":
            session = ort.InferenceSession(onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider', ])
        else:
            session = ort.InferenceSession(onnx, providers=['CPUExecutionProvider', ])
        x = session.get_inputs()[0].name
        y = session.get_outputs()[0].name

        begin = time.time()
        for i, sample_file in enumerate(test_files):
            t = time.time()
            sample_image, shape = self.load_test_data(sample_file, onnx)
            image_path = os.path.join(result_dir, '{0}'.format(os.path.basename(sample_file)))
            fake_img = session.run(None, {x: sample_image})
            self.save_images(fake_img[0], image_path, (shape[1], shape[0]))
            print(f'Processing image: {i}, image size: {shape[1], shape[0]}, ' + sample_file,
                  f' time: {time.time() - t:.3f} s')
        end = time.time()
        print(f'Average time per image : {(end - begin) / len(test_files)} s')

    def ConvertImage(self, img_path, output_path, onnx="model.onnx", device="cpu"):
        result_dir = output_path
        self.check_folder(result_dir)
        if ort.get_device() == 'GPU' and device == "gpu":
            session = ort.InferenceSession(onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider', ])
        else:
            session = ort.InferenceSession(onnx, providers=['CPUExecutionProvider', ])
        x = session.get_inputs()[0].name
        y = session.get_outputs()[0].name

        t = time.time()
        sample_image, shape = self.load_test_data(img_path, onnx)
        image_path = os.path.join(result_dir, '{0}'.format(os.path.basename(img_path)))
        fake_img = session.run(None, {x: sample_image})
        self.save_images(fake_img[0], image_path, (shape[1], shape[0]))
        print(f'Processing image: image size: {shape[1], shape[0]}, ' + img_path, f' time: {time.time() - t:.3f} s')


if __name__ == '__main__':
    onnx_file = 'models/AnimeGANv3_tiny_Cute.onnx'
    input_imgs_path = './assets/test'
    output_path = './assets/output/'
    cs = ConvertStyle()
    cs.ConvertImage("./assets/test/1.jpg", output_path, onnx_file, device='cpu')
    cs.Convert(input_imgs_path, output_path, onnx_file, device='cpu')
