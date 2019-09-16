
import scipy.misc
import  numpy as np
 
def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img
 
content_image = imread('222.png')  # 读取content图片
print(content_image.shape)  #原图shape
 
content_image1 = scipy.misc.imresize(content_image, (200,500))
print(content_image1.shape)  #输入固定大小调整shape
 
content_image2 = scipy.misc.imresize(content_image, 0.5)
print(content_image2.shape)   #输入比例调整shape
 
content_image3 = scipy.misc.imresize(content_image, 25)
print(content_image3.shape)   #输入一个大于1的数，默认为百分比。例如输入5就等于5%
plt.imshow(content_image1) # 显示图片
plt.show()
plt.imshow(content_image2) # 显示图片
plt.show()
plt.imshow(content_image3) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()