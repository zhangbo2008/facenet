
'''
文档:https://pypi.org/project/pysolr/3.3.0/
localhost后面没有.com
'''
from __future__ import print_function
import csv
import csv,json
solr = pysolr.Solr('http://localhost:8080/solr/zhangbo', timeout=10)

#先都删了
solr.delete(q='*:*')




#放入y

b=csv.reader(open(r'C:\Users\张博\Desktop\展示\y.csv'))

b=list(b)
c=b[0]
keys=c
import pysolr
out=[]
print('kaishi' )




j=1
while j <len(b) :
    print('当前放入第%s个到%s+1000)个'%(j,j))
    tmp2=b[j:j+1000]
    for i in tmp2:
        tmp=i
        out.append(dict(zip(keys,tmp)))
        
        
        # How you would index data.
    solr.add(out)    
        
        


        
    j+=1000
    out=[]
print('都放完了')





#放入yhat

b=csv.reader(open(r'C:\Users\张博\Desktop\展示\yhat.csv'))

b=list(b)
c=b[0]
keys=c
import pysolr
out=[]
print('kaishi' )




j=1
while j <len(b) :
    print('当前放入第%s个到%s+1000)个'%(j,j))
    tmp2=b[j:j+1000]
    for i in tmp2:
        tmp=i
        out.append(dict(zip(keys,tmp)))
        
        
        # How you would index data.
    solr.add(out)    
        
        


        
    j+=1000
    out=[]
print('都放完了')













import pysolr
# Setup a basic Solr instance. The timeout is optional.



##a=json.dumps( out ,indent=4)
##print(a)
##
##










import pysolr
#这个url很重要，不能填错了
##solr = pysolr.Solr('http://localhost:8080/solr/jcg/', timeout=10)


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




###搜索jcg中的全部数据
##results = solr.search('*:*')
##
#####搜索id为doc_1的数据
####doc1 = solr.search('id:doc_1')
##
##
##
##
##
##
##
##'''
###删除id为doc_1的数据
##solr.delete(id='doc_1') 
##
###删除所有数据
##solr.delete(q='*:*')
##'''
##
##
##
##
##
##
##
##






