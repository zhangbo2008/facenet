
# -*- coding: utf-8 -*-
import align.detect_face

import os
import facenet
import tensorflow as tf 


base_img_path=os.path.abspath(os.path.join(os.getcwd(), ".."))
base_img_path=os.path.abspath(os.path.join(base_img_path, ".."))
base_img_path=os.path.abspath(os.path.join(base_img_path, "data"))
base_img_path=os.path.abspath(os.path.join(base_img_path, "images"))
base_img_path=os.path.abspath(os.path.join(base_img_path, "Anthony_Hopkins_0001.jpg"))

def load_imgs(img_path = base_img_path,use_to_save = True):
    minsize = 20
    threshold = [0.6,0.7,0.7]
    factor = 0.709
    gpu_memory_fraction=1
    result = {}
    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    img_paths = os.listdir(img_path)
    for image in img_paths:
        if image == '.DS_Store':
            continue
        aligned = mtcnn(os.path.join(img_path, image),minsize,pnet,rnet,onet,threshold,factor)
    
        if aligned is None:
            img_paths.remove(image)
            continue
        if use_to_save:
            result[image.split('.')[0]] = aligned
        else:
            prewhitened = facenet.prewhiten(aligned)  # 鍥剧墖杩涜鐧藉寲
            result[image.split('.')[0]] = prewhitened
    return result
from scipy import misc   #鍥惧儚澶勭悊妯″潡
import numpy as np

# print (os.path.abspath(os.path.join(os.getcwd(), ".."))) #鑾峰彇褰撳墠鐨勪笂绾х洰褰�



def mtcnn(img_path,minsize, pnet, rnet, onet, threshold,factor):
    margin=2
    img = misc.imread(img_path, mode='RGB')  # 璇诲彇鍥剧墖灏嗗浘鐗囪浆鎹㈡垚鐭╅樀
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold,
                                                      factor)  # 鍒╃敤dect_face妫�娴嬩汉鑴�
    # 杩欓噷鐨刡ounding_boxes瀹炶川涓婃槸鎸囧洓涓偣 鍥涗釜鐐硅繛璧锋潵鏋勬垚涓�涓
    if len(bounding_boxes) < 1:
        print("can't detect face, remove ", img_path)  # 褰撹瘑鍒笉鍒拌劯鍨嬬殑鏃跺��,涓嶄繚鐣�
        return None
        # bounding_boxes = np.array([[0, 0, img_size[0], img_size[1]]])
    det = np.squeeze(bounding_boxes[0, 0:4])
    # 杩欓噷鏄负妫�娴嬪埌鐨勪汉鑴告鍔犱笂杈圭晫
    #margin琛ㄧず鐨勫惈涔夋槸鎴戣瘑鍒汉鑴哥殑妗�,璺熸垜瑕佽鍓殑妗嗕箣闂寸殑闂磋窛.
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - margin / 2, 0)
    bb[1] = np.maximum(det[1] - margin / 2, 0)
    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    # 鏍规嵁浜鸿劯妗嗘埅鍙杋mg寰楀埌cropped
    cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    #cropped琛ㄧず鍒囧畬涔嬪悗鐨勫浘鍍�
    # 杩欓噷鏄痳esize鎴愰�傚悎杈撳叆鍒版ā鍨嬬殑灏哄
    #aligned鏄垏瀹屽悗鍐嶆斁缂╁悗鐨勫浘鍍�
    image_size=48
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    return aligned

print(load_imgs())

