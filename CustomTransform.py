import os
from idna import valid_contextj
import litdata as ld
import test

# read data 
train_dataset = ld.StreamingDataset("fast_train_data", shuffle=True, drop_last=True)
valid_dataset = ld.StreamingDataset("fast_valid_data", shuffle=True, drop_last=True)
test_dataset = ld.StreamingDataset("fast_test_data", shuffle=True, drop_last=True)

train_dataloader = ld.StreamingDataLoader(train_dataset, batch_size=32)
valid_dataloader = ld.StreamingDataLoader(valid_dataset, batch_size=32)
test_dataloader = ld.StreamingDataLoader(test_dataset, batch_size=32)   

for i, data in enumerate(train_dataloader):
    print(data)
    if i == 0:
        break

