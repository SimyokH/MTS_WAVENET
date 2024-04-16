# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:15:10 2024

@author: Administrator
"""

class MultiWaveTaskNet(nn.Module):
    def __init__(self):
        super(MultiWaveTaskNet, self).__init__()
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.x_dim = model_args["x_dim"]
        self.num_layer = model_args["num_layer"]
        self.if_spatial = model_args["If_spatial"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.input_dim = model_args["input_dim"]
        self.drop_out_rate = model_args["Drop_out_rate"]
        
        self.fc1 = nn.Linear(7, 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128,64)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=self.drop_out_rate)

        #using the same embedding dimension
        self.node_emb = nn.Embedding(num_embeddings=1, embedding_dim=self.node_dim)
        self.hs_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.x_dim)
        self.T_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.x_dim)
        self.Dir_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.x_dim)

        # final-regression
        self.regression_layer = nn.Linear(64,3, bias=True)
        self.regression_layer1 = nn.Linear(33,1, bias=True)
        self.regression_layer2 = nn.Linear(33,1, bias=True)
        self.regression_layer3 = nn.Linear(33,1, bias=True)


    def forward(self, in_data,additional_feature):
        in_data = in_data.unsqueeze(1)
        in_data = in_data.repeat(1,self.num_nodes,1)
        
        additional_feature = additional_feature.unsqueeze(0)
        additional_feature = additional_feature.repeat(len(in_data),1,1)
        
        #variables embedding
        hs_indx = torch.Tensor([list(range(self.num_nodes)) for _ in range(len(in_data))]).long().to(self.device)
        hs_emb = []
        if self.if_spatial:
            hs_emb.append(
                self.hs_emb(hs_indx)
            )
        hs_emb = torch.Tensor(np.array([item.detach().cpu().numpy() for item in hs_emb]))
        hs_emb = hs_emb.squeeze(0)
        
        T_indx = torch.Tensor([list(range(self.num_nodes)) for _ in range(len(in_data))]).long().to(self.device)
        T_emb = []
        if self.if_spatial:
            T_emb.append(
                self.T_emb(T_indx)
            )
        T_emb = torch.Tensor(np.array([item.detach().cpu().numpy() for item in T_emb]))
        T_emb = T_emb.squeeze(0)
        
        Dir_indx = torch.Tensor([list(range(self.num_nodes)) for _ in range(len(in_data))]).long().to(self.device)
        Dir_emb = []
        if self.if_spatial:
            Dir_emb.append(
                self.Dir_emb(Dir_indx)
            )
        Dir_emb = torch.Tensor(np.array([item.detach().cpu().numpy() for item in Dir_emb]))
        Dir_emb = Dir_emb.squeeze(0)
        
        hidden_with_additional_feature = torch.cat([in_data,additional_feature],dim=2)
        
        # vanilla wavenet
        x = self.act(self.fc1(hidden_with_additional_feature))
        x = self.act(self.fc2(self.drop(x)))
        x = self.act(self.fc3(self.drop(x)))
        x = x.squeeze(2)
        
        outputfinal = self.regression_layer(x)
        prehs_t = outputfinal[:,:,0].unsqueeze(2)
        preT_t = outputfinal[:,:,1].unsqueeze(2)
        preDir_t = outputfinal[:,:,2].unsqueeze(2)
        hs_with_emb = torch.cat([prehs_t,hs_emb],dim=2)
        T_with_emb = torch.cat([preT_t,T_emb],dim=2)
        Dir_with_emb = torch.cat([preDir_t,Dir_emb],dim=2)
        
        #res-connet &final regression
        prehs = self.regression_layer1(hs_with_emb)+prehs_t
        preT = self.regression_layer2(T_with_emb)+preT_t
        preDir = self.regression_layer3(Dir_with_emb)+preDir_t
        

        prehs = prehs.squeeze()
        preT = preT.squeeze()
        preDir = preDir.squeeze()

        return prehs,preT,preDir