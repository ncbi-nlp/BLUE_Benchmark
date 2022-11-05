import pandas as pd
import os

filename = '../../example_dataset/data/test.tsv'
path = os.path.join(os.getcwd(), filename)
print(path)
df = pd.read_csv(filepath_or_buffer= path ,sep= '\t',header= 0)
def show(df, num):
    print(df.head(num))
def write_tsv(buff, header, saving_path):
    with open(saving_path,'wb') as f:
        f.write(buff.to_csv(sep='\t', index= header))


# print(df.info())

# s1 means sentence 1 and s2 means sentence 2
data2 = pd.read_excel(io= "pairs.xls",header=0)
data2.columns= ["index","s1",'s2']
data2['index'] = [i for i in range(data2.count()[0])]
# data2.reset_index()
# print(data2['index'])
# print(data2.count()[0])
# print(data2.loc[data2.index==1])
# print(data2)
score_file = 'scores.xls'
score_path = os.path.join(os.getcwd(),score_file)
score = pd.read_excel(io=score_path,header=0)

score.columns = ['index', 'score1','score2','score3','score4','score5']
score['index'] = [i for i in range(score.count()[0])]

print(score[score.columns[1:]].mean(axis=1))
col = ['index', 'genre', 'filename', 'year', 'old_index', 'source1', 'source2', 'sentence1', 'sentence2',
                 'score']
new = pd.DataFrame(index=score['index'],columns= col)
new['index'] = score.index

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test  = train_test_split(new.index,new.index,test_size= 0.1,random_state= 0)
# print(len(x_test))
# print(len(x_train))


new = new.sample(frac=1.0)
print(new.drop(columns='index'))
print(new.reset_index(drop=True))
print(new.iloc[0:5])