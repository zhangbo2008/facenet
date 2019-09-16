# -*- coding: utf-8 -*-


'''
函数.__name__表示这个函数的名字
'''

def main(a):
    print(a.__name__)
def main2():
    return 1
(main(main2))
