def debug(func):
    def wrapper():
        print ("[DEBUG]: enter {}()".format(func.__name__))
        return func()
    return wrapper

@debug
def say_hello():
    print ("hello!")
say_hello()

'''
从上面例子中可以看出来debug这个嵌套函数里面的
小函数就是最后包装后的函数的运行内容.

加上@装饰器就能实现这个功能了!
'''
    
    
