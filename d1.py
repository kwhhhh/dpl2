import torch
import os
import pandas as pd

os.makedirs(os.path.join('..','data'), exist_ok=True)
data_file = os.path.join('..','data','house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

data = pd.read_csv(data_file)

inputs, outputs = data.iloc[:,0:2],data.iloc[:,2]
inputs = inputs.fillna(inputs.mean(numeric_only = True))
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
print(inputs)

X,y = torch.tensor(inputs.values,dtype=float), torch.tensor(outputs.values,dtype=float)
print(X,y)
# x = torch.arange(12)
# y = torch.zeros((1,2,3))
# z = torch.tensor([[[1,2,3],[1,1,3],[3,2,4]]])

# print(x)
# print(x.shape)
# print(x.numel())
# print(y)
# print(z.shape)
# print(x.sum())

