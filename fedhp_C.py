"""Implementation of the FedHP: Federated Learning with Hyperspherical Prototypical Regularization
[FedHP24]_ algorithm.

References:
    .. [FedHP24] Samuele Fonio, Mirko Polato, Roberto Esposito.
       FedHP: Federated Learning with Hyperspherical Prototypical Regularization. In ESANN (2024).
       URL: https://www.esann.org/sites/default/files/proceedings/2024/ES2024-183.pdf
"""
import copy
import sys
from typing import Collection, Generator, Literal, Any
import os
import torch
import torch.optim as optim
from rich.progress import track
from torch import nn
from torch.optim.optimizer import Optimizer as Optimizer
from copy import deepcopy

sys.path.append(".")
sys.path.append("..")
from fluke import FlukeENV  # NOQA
from fluke.client import Client  # NOQA
from fluke.comm import Message  # NOQA
from fluke.config import OptimizerConfigurator  # NOQA
from fluke.data import FastDataLoader  # NOQA
from fluke.evaluation import Evaluator  # NOQA
from fluke.server import Server  # NOQA
from fluke.utils import clear_cuda_cache  # NOQA
from fluke.utils.model import get_activation_size, unwrap  # NOQA
from fluke.algorithms import CentralizedFL, PersonalizedFL  # NOQA

from fluke.algorithms.fedhp import FedHPClient, FedHPModel, FedHPServer, ProtoNet, SeparationLoss  # NOQA

class FedHP_C_Client(FedHPClient):
    
    @torch.no_grad()
    def receive_model(self) -> None:
        if self.anchors is None:
            msg = self.channel.receive(self.index, "server", msg_type="anchors")
            self.anchors = msg.payload.data
        
        msg = self.channel.receive(self.index, "server", msg_type="prototypes")
        # here the prototypes must be a dictionary (or whatsoever) 
        # of all the prototypes registered on the server
        global_prototypes = msg.payload
        local_prototypes = deepcopy(self.model.prototypes.data)
        if self._last_round > 1:
            cosine_similarity = torch.nn.CosineSimilarity(dim=1)
            max_similarity_index = float('-inf')
            # possibility: for each label choose the one the prototype that deviate less
            for ind, prototypes in enumerate(global_prototypes):
                similarity = torch.mean(cosine_similarity(local_prototypes, prototypes))
                if similarity.item() > max_similarity_index:
                    max_similarity_index = similarity.item()
                    new_prototypes = deepcopy(prototypes)
                    self.model.prototypes.data = new_prototypes

    def fit(self, override_local_epochs: int = 0) -> float:
        epochs: int = (
            override_local_epochs if override_local_epochs > 0 else self.hyper_params.local_epochs
        )
        self.model.train()
        self.model.to(self.device)

        def filter_fun(model):
            return [param for name, param in model.named_parameters() if "prototype" not in name]

        if self.optimizer is None:
            self.optimizer, self.scheduler = self._optimizer_cfg(self.model, filter_fun=filter_fun)

        if self.proto_opt is None:
            proto_params = [p for name, p in self.model.named_parameters() if "proto" in name]
            self.proto_opt = optim.Adam(proto_params, lr=0.005)

        running_loss = 0.0
        for _ in range(epochs):
            for _, (X, y) in enumerate(self.train_set):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                self.proto_opt.zero_grad()
                _, dists = self.model.forward(X)
                loss = (1 - self.hyper_params.lam) * self.hyper_params.loss_fn(dists, y)
                loss_proto = torch.mean(
                    1 - nn.CosineSimilarity(dim=1)(unwrap(self.model).prototypes, self.anchors)
                )
                loss += self.hyper_params.lam * loss_proto
                loss.backward()
                self._clip_grads(self.model)
                self.optimizer.step()
                self.proto_opt.step()
                running_loss += loss.item()
            self.scheduler.step()
        running_loss /= epochs * len(self.train_set)
        self.model.cpu()
        clear_cuda_cache()
        return running_loss
        
    def send_model(self) -> None:
        self.channel.send(Message(copy.deepcopy(self.model.prototypes.data),
                          "prototypes", self.index, inmemory=True), "server")

class FedHP_C_Server(Server):
    def __init__(self,
                 model: nn.Module,
                 test_set: FastDataLoader,
                 clients: Collection[Client],
                 weighted: bool = True,
                 n_protos: int = 10,
                 embedding_size: int = 100,
                 soft: bool = False,
                 K: int = 9):
        super().__init__(model=ProtoNet(model, n_protos, embedding_size),
                         test_set=None,
                         clients=clients,
                         weighted=weighted)
        self.hyper_params.update(n_protos=n_protos,
                                 embedding_size=embedding_size,
                                 soft=soft, 
                                 K=K)

        self.device = FlukeENV().get_device()
        self.anchors = None
        self.prototypes = None
        # self.clients_class_weights = None
        self.temp_protos = None

    def fit(self, n_rounds: int = 10, eligible_perc: float = 0.1, finalize: bool = True) -> None:
        
        if self.rounds == 0:
            self.id_exp = -1
            for obs in self._observers:
                if isinstance(obs, CentralizedFL):
                    self.id_exp = obs.id
            print(f"ID EXP: {self.id_exp}")
            self.anchors = self._hyperspherical_embedding().data
            self.channel.broadcast(Message(self.anchors, "anchors", "server"),
                                   [c.index for c in self.clients])
            self.prototypes = copy.deepcopy(self.anchors)

        return super().fit(n_rounds=n_rounds, eligible_perc=eligible_perc, finalize=finalize)

    def _hyperspherical_embedding(self):
        """
        Function to learn the prototypes according to the ``SeparationLoss`` minimization.
        """
        lr = 0.1
        momentum = 0.9
        n_steps = 1000
        wd = 1e-4
        # torch.manual_seed(seed)
        mapping = torch.rand((self.hyper_params.n_protos, self.hyper_params.embedding_size),
                             device=self.device, requires_grad=True)
        optimizer = torch.optim.SGD([mapping], lr=lr, momentum=momentum, weight_decay=wd)
        loss_fn = SeparationLoss()
        for _ in track(range(n_steps), "[SERVER] Learning prototypes..."):
            with torch.no_grad():
                mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
            optimizer.zero_grad()
            loss = loss_fn(mapping)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            mapping.div_(torch.norm(mapping, dim=1, keepdim=True))
        return mapping.detach()

    def broadcast_model(self, eligible: Collection[Client]) -> None:
        if self.temp_protos is None:
            self.temp_protos = [self.prototypes for j in range(len(eligible))]
        # This function broadcasts the prototypes to the clients
        self.channel.broadcast(Message(self.temp_protos, "prototypes", "server"),
                               [c.index for c in eligible])

    def receive_client_models(self,
                              eligible: Collection[Client],
                              state_dict: bool = False) -> Generator[nn.Module, None, None]:
        for client in eligible:
            yield self.channel.receive("server", client.index, "prototypes").payload
    
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
    
    def aggregate(self,
                  eligible: Collection[FedHPClient],
                  client_models: Collection[nn.Module]) -> None:
        clients_protos = list(client_models)
        sim_scores = self._compute_similarity(clients_protos)
        os.makedirs(f"results_fedhp/{self.hyper_params['K']}/{self.id_exp}", exist_ok=True)
        torch.save(sim_scores, f"results_fedhp/{self.hyper_params['K']}/{self.id_exp}/sim_scores_{self.rounds}.pt")
        self.temp_protos = []
        weights = self._get_client_weights(eligible)
        weights = torch.FloatTensor(weights)
        if self.hyper_params['soft']:
            
            for i, client_protos in enumerate(clients_protos):

                temp_client_protos = deepcopy(client_protos)
                w = torch.nn.functional.softmax(sim_scores[i,:] * weights)

                temp_client_protos = w * temp_client_protos
                
                temp_client_protos = torch.sum(torch.stack(temp_client_protos), dim=0) / len(clients_protos)

                self.temp_protos.append(temp_client_protos)
        
        else:
            for i, client_protos in enumerate(clients_protos):
                
                temp_client_protos = deepcopy(client_protos)
                nearest_clients_ind = torch.topk(sim_scores[i,:], self.hyper_params['K'] + 1).indices.numpy()
                
                nearest_clients_sd = [clients_protos[j] for j in nearest_clients_ind]
                
                temp_client_protos = torch.sum(torch.stack(nearest_clients_sd), dim=0) / len(nearest_clients_ind)

                self.temp_protos.append(temp_client_protos)

class FedHP_C(PersonalizedFL):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.server.attach(self)

    def get_client_class(self) -> type[Client]:
        return FedHP_C_Client

    def get_server_class(self) -> type[Server]:
        return FedHP_C_Server

