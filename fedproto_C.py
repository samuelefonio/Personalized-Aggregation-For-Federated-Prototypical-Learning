"""Implementation of the FedProto [FedProto22]_ algorithm.
References:
    .. [FedProto22] Yue Tan, Guodong Long, Lu Liu, Tianyi Zhou, Qinghua Lu, Jing Jiang, Chengqi
       Zhang. FedProto: Federated Prototype Learning across Heterogeneous Clients. In AAAI (2022).
       URL: https://arxiv.org/abs/2105.00243
"""
import sys
from collections import defaultdict, OrderedDict
from copy import deepcopy
from typing import Collection, Any
from fluke import FlukeENV

import torch
from torch.nn import Module

sys.path.append(".")
sys.path.append("..")

from fluke.client import Client, PFLClient  # NOQA
from fluke.comm import Message  # NOQA
from fluke.config import OptimizerConfigurator  # NOQA
from fluke.data import FastDataLoader  # NOQA
from fluke.evaluation import Evaluator  # NOQA
from fluke.nets import EncoderHeadNet  # NOQA
from fluke.server import Server  # NOQA
from fluke.utils import clear_cuda_cache, get_model  # NOQA
from fluke.utils.model import unwrap  
from fluke.algorithms import CentralizedFL, PersonalizedFL
import os
from fluke.algorithms.fedproto import FedProtoClient, FedProtoModel, FedProtoServer  # NOQA

def from_dict_to_tensor(d: dict) -> torch.Tensor:
    keys = list(d.keys())
    tensors = [d[k] for k in keys]

    # Stack into a single tensor of shape [B, C, N]
    out_tensor = torch.stack(tensors, dim=0)
    return out_tensor

class FedProto_C_Client(FedProtoClient):
    
    def get_protos(self):
        self.model.train()
        self.model.to(self.device)
        protos = defaultdict(list)
        for _ in range(1):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                
                Z = unwrap(self.model).encoder(X)
                
                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(Z[i, :].detach().data)

        self.model.cpu()
        clear_cuda_cache()
        self._update_protos(protos)

    def receive_model(self) -> None:
        msg = self.channel.receive(self.index, "server", msg_type="model")
        # here the prototypes must be a dictionary (or whatsoever) 
        # of all the prototypes registered on the server
        global_prototypes = msg.payload
        
        if self._last_round > 1:
            global_prototypes = [from_dict_to_tensor(i) for i in global_prototypes]
            if self.prototypes[0] == None:
                self.get_protos()
            self.prototypes = from_dict_to_tensor(self.prototypes)
            cosine_similarity = torch.nn.CosineSimilarity(dim=1)
            max_similarity_index = float('-inf')
            # possibility: for each label choose the one the prototype that deviate less
            for ind, prototypes in enumerate(global_prototypes):
                similarity = torch.mean(cosine_similarity(self.prototypes, prototypes))
                if similarity.item() > max_similarity_index:
                    max_similarity_index = similarity.item()
                    new_prototypes = deepcopy(prototypes)
                    self.global_protos = new_prototypes
            self.global_protos = {i: self.global_protos[i,:] for i in range(self.hyper_params.n_protos)}
            self.prototypes = {i: self.prototypes[i,:] for i in range(self.hyper_params.n_protos)}
        else:
            self.global_protos = global_prototypes[0]

class FedProto_C_Server(Server):

    def __init__(self,
                 model: Module,
                 test_set: FastDataLoader,
                 clients: Collection[Client],
                 weighted: bool = True,
                 n_protos: int = 10,
                 soft: bool = False,
                 K: int = 9):
        super().__init__(model=None, test_set=None, clients=clients, weighted=weighted)
        
        self.hyper_params.update( n_protos=n_protos,
                                 soft=soft, K=K)
        self.prototypes = [None for _ in range(self.hyper_params.n_protos)]
        self.temp_protos = None

        
    def broadcast_model(self, eligible: Collection[Client]) -> None:
        if self.temp_protos is None:
            self.temp_protos = [{i: None for i in range(self.hyper_params.n_protos)} for j in range(len(eligible))]
        # This function broadcasts the prototypes to the clients
        self.channel.broadcast(Message(self.temp_protos, "model", "server"),
                               [c.index for c in eligible])

    def _compute_similarity(self, protos: list) -> float:

        cos = torch.nn.CosineSimilarity(dim=1)

        similarity_matrix = torch.zeros(len(protos), len(protos))

        for i in range(len(protos)):
            for j in range(len(protos)):
                if i == j:
                    # assert sims.mean() == 1.0, f"similarity between corresponding prototypes is not 1, it is {sims.mean()}"
                    similarity_matrix[i, j] = 1.0  
                    continue
                sims = cos(protos[i], protos[j])  
                similarity_matrix[i, j] = sims.mean()  
                similarity_matrix[j, i] = sims.mean()
        return similarity_matrix
    
    def fit(self, *args: Any, **kwargs: Any) -> None:
        
        if self.rounds == 0:
            self.id_exp = -1
            for obs in self._observers:
                if isinstance(obs, CentralizedFL):
                    self.id_exp = obs.id
            print(f"ID EXP: {self.id_exp}")

        return super().fit(*args, **kwargs)
        
    @torch.no_grad()
    def aggregate(self, eligible: Collection[Client], client_models: Collection[Module]) -> None:
        
        clients_protos = [from_dict_to_tensor(i).to(self.device) for i in client_models]
        
        # calculate the average cosine similarity among all the pairs
        sim_scores = self._compute_similarity(clients_protos)
        
        if self.rounds%10 == 0:
            os.makedirs(f"results_fedproto/{self.hyper_params['K']}/{self.id_exp}", exist_ok=True)
            torch.save(sim_scores, f"results_fedproto/{self.hyper_params['K']}/{self.id_exp}/sim_scores_{self.rounds}.pt")
        self.temp_protos = []

        weights = self._get_client_weights(eligible)
        weights = torch.FloatTensor(weights)
        if self.hyper_params['soft']:
            
            for i, client_protos in enumerate(clients_protos):

                temp_client_protos = deepcopy(client_protos)
                w = torch.nn.functional.softmax(sim_scores[i,:] * weights)

                temp_client_protos = w * temp_client_protos
                torch.stack(temp_client_protos) # to verify dimension
                temp_client_protos = torch.sum(torch.stack(temp_client_protos), dim=0) / len(clients_protos)
                temp_client_protos = {j: temp_client_protos[j,:] for j in range(self.hyper_params.n_protos)}
                self.temp_protos.append(temp_client_protos)
        
        else:
            for i, client_protos in enumerate(clients_protos):

                temp_client_protos = deepcopy(client_protos)
                nearest_clients_ind = torch.topk(sim_scores[i,:], self.hyper_params['K'] + 1).indices.numpy()

                nearest_clients_sd = [clients_protos[j] for j in nearest_clients_ind]
                
                temp_client_protos = torch.sum(torch.stack(nearest_clients_sd), dim=0) / len(nearest_clients_ind)

                temp_client_protos = {j: temp_client_protos[j,:] for j in range(self.hyper_params.n_protos)}
                print({torch.norm(temp_client_protos[j]).item() for j in range(self.hyper_params.n_protos)})
                self.temp_protos.append(temp_client_protos)
        

class FedProto_C(PersonalizedFL):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.server.attach(self)

    def get_client_class(self) -> PFLClient:
        return FedProto_C_Client

    def get_server_class(self) -> type[Server]:
        return FedProto_C_Server
