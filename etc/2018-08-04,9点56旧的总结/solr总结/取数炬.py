#从solr中读取jason格式的数据



import pysolr
#这个url很重要，不能填错了
##solr = pysolr.Solr('http://localhost:8080/solr/jcg/', timeout=10)
solr = pysolr.Solr('http://localhost:8080/solr/index.html#/zhangbo/documents', timeout=10)
print(423)

'''
插入:
'''

#正确的数据格式，可以少项
##solr.add([
##    {
##        "id": "doc_1",
##        "name": "A test document",
##        "cat": "book",
##        "price": "7.99",
##        "inStock": "T",
##        "author": "George R.R. Martin",
##        "series_t": "A Song of Ice and Fire",
##        "sequence_i": "1",
##        "genre_s": "fantasy",
##    }
##])





'''
取数据


'''
solr = pysolr.Solr('http://localhost:8080/solr/zhangbo', timeout=10)

#一定要在这个地址的solr上操作.
solr.delete(q='*:*')
print(324234)
solr.add([
    {
        "idd": "doc_1",
        "title": "A",
        
    },
    {
        "idd": "doc_2",
        "title": "B",
        
    },
    {
        "idd": "doc_1",
        "title": "C",
        
    },
])


#打印全部
print(list(solr.search('*:*')))

'''

#测试了一天,可算试出来如何高级搜索了.遇到问题可以看这个py库的源码.

#这个search函数的原理是**kwarg传入一个字典.字典的key是查询的命令,

#value是你要查询命令的输入框中应该写入的内容.也是从csdn

https://blog.csdn.net/sinat_33455447/article/details/63341339

这个网页受到的启示.他写的是start后面接页数,所以对应改fq,后面就是接一个条件.



'''     

doc1 = solr.search("idd:doc_1" , **{"fq":'title:A'})



#读取之后需要list一下,会自动去掉垃圾信息
doc1=list(doc1)

print(doc1)




# Setup a basic Solr instance. The timeout is optional.



print('打印完了')

