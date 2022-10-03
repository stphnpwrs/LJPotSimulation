import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from os import listdir
from os.path import join, isfile
import pickle
import torch_geometric

from torch_geometric.data import Data, InMemoryDataset, DataLoader

class LJDataset(InMemoryDataset):
    
    def __init__(self, root, transform=None, pre_transform=None):
        super(LJDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    
    @property
    def processed_file_names(self):
        return ['processed_data.dataset']
    
    def download(self):
        # Download to `self.raw_dir`.
        pass
    
    def process(self):

        data_list = []
        
        files = listdir('./logs')
        count = 0
        for i in range(len(files)):
            f = files[i]
            filepath = join('./logs/',f)
            if isfile(filepath):
            
                if count == 1000:
                    break
                count += 1
            
                print("File: " + str(i))
                
                try:
                    fileData = pickle.load(open(filepath, 'rb'))
                except EOFError as e:
                    print(filepath)
                        
                traj = fileData['traj']
                nHis = fileData['nHis']
                acc = fileData['acc']
                
                nAgents = len(traj)      # 30 agents
                nSteps = len(traj[0])    # 100 steps (10 seconds)

                for j in range(nSteps-1):

                    d1 = [] # current positions
                    d2 = [] # plus one step
                    e1 = [] # edge from
                    e2 = [] # edge to
                    
                    d11 = []
                    d12 = []
                    d21 = []
                    d22 = []
                    
                    att = []

                    for k in range(nAgents):                        
                        
                        v1 = traj[k][j][0]
                        v2 = traj[k][j][1]
                        d1.append([v1, v2])
                        d11.append(v1)
                        d12.append(v2)
                        
                        v3 = traj[k][j+1][0]
                        v4 = traj[k][j+1][1]
                        d2.append(acc[k][j])
                        
                        e1.extend([k for _ in range(len(nHis[k][j]))])
                        for val in nHis[k][j]:
                            try:
                                e2.append(val[0])
                                att.append(val[1])
                            except TypeError as e:
                                e2.append(val-1)
                                att.append([1])

                    e = []
                    for k in range(len(e1)):
                        e.append([e1[k], e2[k]])
                    
                    x = torch.tensor(d1, dtype=torch.float)
                    e = torch.tensor(e, dtype=torch.long)
                    e = torch.tensor([e1, e2], dtype=torch.long)
                    a = torch.tensor(att, dtype=torch.float)
                    y = torch.tensor(d2, dtype=torch.float)
                    
                    data_list.append(Data(x=x, edge_index=e, edge_attr=a, y=y))
                

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
               