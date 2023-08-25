import os
f = os.listdir('D:\최종본 웹코드\static')
print(f)
vlad=[]
count='ⓠ'
for i in f:
    if "wav" in i:
        print(i)
        vlad.append(i)
if len(vlad)==8:
    pass
else:
    while 1:
        vlad.append(count)
        if len(vlad)==8:
            break
print(vlad)
