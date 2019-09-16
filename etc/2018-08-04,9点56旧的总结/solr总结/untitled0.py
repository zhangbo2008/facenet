# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:59:16 2018

@author: 张博
"""

while 1:
    try :
        
        tmp=int(input())
        list1=[int(i)for i in input().split()]
        a=sorted(list1)
        i=0
        j=0
        count=0
        while i<len(list1):
            tmp1=list1[i]
            tmp2=a[j]
            if tmp1!=tmp2:
                count+=1
                i+=1
            else:
                i+=1
                j+=1
        print(count)
                
            
        
    except:
        break