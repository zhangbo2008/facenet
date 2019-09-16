https://github.com/davidsandberg/facenet

把这个项目的所有最后用到的.py都写在这里.



src/face/facenet-master/src/use_facenet_compare_pics.py
这个文件51行传入参数a,b,c表示3个图片的地址.后面的参数160是不能变动的.网络里面的接受参数160写死的.输出矩阵叫做Distance matrix表示3个图片之间分别的距离.每一个个图片看做了高维空间中的一点

下面需要看如何用tf进行fine tune.


使用代码问题不大,关键是看源码如何实现,看能不能做改动.



2019-01-14,17点55
通过KDtree对摄像头采集人脸的本地图片embed之后的全部矩阵放入KDTree结构中
从而用logn的效率来找最接近摄像头的人脸图片和对应的距离.

.py文件的说明:

face\facenet-master\contributed\use_real_time_face_recognition.py  
摄像头识别人脸,首先把即将被比较的图片都计算出embed,存入KDTree中.这样在图片
数据集大的时候平均logn的复杂度就能找到最接近摄像头里面的人脸和他们之间的距离.
之后比较L2距离如果大于1.1就判定是同一张脸,标注图片的文件名给摄像头上的人脸的
一角,否则就写Unknown.

face\facenet-master\src\use_facenet_compare_pics.py
里面main函数第一个参数是一个图片地址,第二个参数是文件夹地址.函数返回的是
图片和文件夹中所有文件的距离组成的数组

face\facenet-master\src\search_the_most_similar_pic.py
里面main函数第一个参数是一个图片地址,第二个参数是文件夹地址.函数返回的是
一个文件夹中的图片,这个图片跟第一个参数表示的图片距离最近


face\facenet-master\src\use_for_align_data_mtcnn.py
main函数第一个参数是存很多原始图片的文件夹source
,第二个参数是存裁剪后的图片的文件夹obj
.函数运行后source中的图片如果有人脸就进行裁剪人脸放入obj文件中.否则不处理




最后利用kdtree加速搜索.高维向量



