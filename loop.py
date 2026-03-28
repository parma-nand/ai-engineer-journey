li=[1,5,4,2]
for i in range(len(li)):
    print(li[i])

tup = ("geeks", "for", "geeks")
for x in tup:
    print(x)
    
s = "abc"
for x in s:
    print(x)
    
d = dict({'x':123, 'y':354})
for x in d:
    print("%s  %d" % (x, d[x]))
    
a={'a':10,'v':12,'f':13}
for i,(k,v) in enumerate(a.items()):
    print(i,k,v)