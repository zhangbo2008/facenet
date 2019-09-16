
'''
一个叫compare.py文件

__name__:表示compare这个字符
__doc__:表示这个文件最开始的3引号注释  里面的内容一般是这个.py文件的用途说明


__package__:使用的包
'''
from __future__ import absolute_import
print(__loader__.path)
print(__loader__.name)
print(__package__)
print(__doc__)
print(__spec__)
print(__builtins__.__name__)



'''
关于这句from __future__ import absolute_import的作用: 
直观地看就是说”加入绝对引入这个新特性”。说到绝对引入，当然就会想到相对引入。那么什么是相对引入呢?比如说，你的包结构是这样的: 
pkg/ 
pkg/init.py 
pkg/main.py 
pkg/string.py

            #如果你在main.py中写import string,那么在Python 2.4或之前, Python会先查找当前目录下有没有string.py, 若找到了，则引入该模块，然后你在main.py中可以直接用string了。如果你是真的想用同目录下的string.py那就好，但是如果你是想用系统自带的标准string.py呢？那其实没有什么好的简洁的方式可以忽略掉同目录的string.py而引入系统自带的标准string.py。这时候你就需要from __future__ import absolute_import了。这样，你就可以用import string来引入系统的标准string.py, 而用from pkg import string来引入当前目录下的string.py了
--------------------- 
作者：caiqiiqi 
来源：CSDN 
原文：https://blog.csdn.net/caiqiiqi/article/details/51050800 
版权声明：本文为博主原创文章，转载请附上博文链接！
'''
print(type(absolute_import))




print(absolute_import.mandatory)































