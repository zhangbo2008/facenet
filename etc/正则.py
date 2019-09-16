import re

a=re.match(r'(^model-[\w\- ]+.ckpt-(\d+))','model-20170512-110547.ckpt-250000.data-00000-of-00001')
print(a.groups())