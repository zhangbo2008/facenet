def fmap(a, b):
        return (a, b)
lik = range(1, 11)
liv = list("abcdefghij")
lim = map(fmap, lik, liv)
d = dict(lim)
print( d)