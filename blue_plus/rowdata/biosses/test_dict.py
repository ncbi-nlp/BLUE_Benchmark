import pandas as pd

index = [ 2,4,8,9,10]
col = ['index','B']

da = pd.DataFrame(data=None, columns= col)

da['index'] = index
da = da.reindex()
da = da.sample(frac=1.0)

print(da)

da = da.reset_index(drop=True)

for i in da['index']:
    print(i)

da.to_csv('sssss.tsv',sep='\t')