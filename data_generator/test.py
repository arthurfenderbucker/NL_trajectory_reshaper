def generator2():
    for i in range(10):
        yield i*10

def generator3():
    for j in range(10):
        yield j

def generator(a, x = []):
    if isinstance(a, tuple):
        for i,y in generator(a[0], x=x):
            print("i",i)
            print("i",y)
            for j,z in generator(a[1],x=x):
                # print("j:", j)
                yield (i + j, x)
    if isinstance(a, list):
        for i in a:
            yield (i,None)
        # yield from a
    if isinstance(a, dict):
        # c_value = a.values()
        # print(c_value)

        for c_text,c_value in a.items():
            yield (c_value["value"], None)
            # c_text,c_value  = random.choice(list(p.items()))
        # yield from c_value

    # for i in generator2():
    #     yield i
    # for j in generator3():
    #     yield j
b = {"a":{"value":100},"b":{"value":200}}
a = ([1,2,3],b)
# a = [1,2,3]
for i,j in generator(a):
    print("==>",i, " --- ",j)
for i,j in [("",None)]:
    print("==>",i, " --- ",j)

from itertools import chain

def generator():
    for v in chain(generator2(), generator3()):
        yield v