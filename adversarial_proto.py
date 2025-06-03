
import sys
from typing import Any

import torch
import copy
sys.path.append(".")
sys.path.append("..")

from fluke.client import PFLClient  # NOQA
from fluke.comm import Message  # NOQA
from fluke.server import Server  # NOQA
from fluke.utils import clear_cuda_cache, get_model  # NOQA
from fluke.utils.model import unwrap  
from fluke.algorithms import PersonalizedFL
import os
from fluke.algorithms.fedproto import FedProtoClient, FedProtoServer  # NOQA
from fluke.algorithms.fedhp import FedHPServer, FedHPClient  # NOQA
from fedprotoIFCA import FedProtoIFCAClient, FedProtoIFCAServer  # NOQA
from fedhpIFCA import FedHPIFCAClient, FedHPIFCAServer  # NOQA
from rich.progress import track

class FedProtoClient_Adv(FedProtoClient):
    
    def __init__(self, malicious_percentage=0, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.malicious = torch.rand(1).item() < malicious_percentage
        print(self.malicious)

    def fit(self, override_local_epochs = 0):
        if self.malicious and self.prototypes[0] == None:
            return super().fit(1)
        else:
            return super().fit(override_local_epochs)

    def send_model(self) -> None:
        if self.malicious:
            print(f'client {self.index} is malicious, sending random prototypes')
            self.channel.send(Message({i: torch.rand(self.prototypes[i].shape[0]) for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
        else:    
            self.channel.send(Message(self.prototypes, "model", self.index, inmemory=True), "server")

class FedProtoServer_Adv(FedProtoServer):

    def finalize(self):
        if self.rounds == 0:
            return
        client_to_eval = [client for client in self.clients if client.index in self._participants]
        self.broadcast_model(client_to_eval)
        for client in track(client_to_eval, "Finalizing federation...", transient=True):
            if client.malicious:
                print(f"Client {client.index} is malicious, skipping finalization.")
                continue
            client.finalize()
        # self._compute_evaluation(self.rounds, client_to_eval)
        self.notify(event="finished", round=self.rounds + 1)

class FedProtoIFCAClient_Adv(FedProtoIFCAClient):
    
    def __init__(self, malicious_percentage=0, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.malicious = torch.rand(1).item() < malicious_percentage
        print(self.malicious)

    def fit(self, override_local_epochs = 0):
        if self.malicious and self.prototypes[0] == None:
            return super().fit(1)
        else:
            return super().fit(override_local_epochs)

    def send_model(self) -> None:
        if self.malicious:
            print(f'client {self.index} is malicious, sending random prototypes')
            self.channel.send(Message({i: torch.rand(self.prototypes[i].shape[0]) for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
        else:    
            self.channel.send(Message(self.prototypes, "model", self.index, inmemory=True), "server")

class FedProtoIFCAServer_Adv(FedProtoIFCAServer):

    def finalize(self):
        if self.rounds == 0:
            return
        client_to_eval = [client for client in self.clients if client.index in self._participants]
        self.broadcast_model(client_to_eval)
        for client in track(client_to_eval, "Finalizing federation...", transient=True):
            if client.malicious:
                print(f"Client {client.index} is malicious, skipping finalization.")
                continue
            client.finalize()
        # self._compute_evaluation(self.rounds, client_to_eval)
        self.notify(event="finished", round=self.rounds + 1)


class FedHPClient_Adv(FedHPClient):
    
    def __init__(self, malicious_percentage=0, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.malicious = torch.rand(1).item() < malicious_percentage
        print(self.malicious)

    def fit(self, override_local_epochs = 0):
        if self.malicious:
            return super().fit(1)
        else:
            return super().fit(override_local_epochs)

    def send_model(self) -> None:
        if self.malicious:
            print(f'client {self.index} is malicious, sending random prototypes')
            # self.channel.send(Message({i: torch.rand(self.prototypes[i,:].shape) for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
            self.channel.send(Message(torch.rand(self.model.prototypes.data.shape),"prototypes", self.index, inmemory=True), "server")
        else:    
            self.channel.send(Message(copy.deepcopy(self.model.prototypes.data),"prototypes", self.index, inmemory=True), "server")


class FedHPServer_Adv(FedHPServer):

    def finalize(self):
        if self.rounds == 0:
            return
        client_to_eval = [client for client in self.clients if client.index in self._participants]
        self.broadcast_model(client_to_eval)
        for client in track(client_to_eval, "Finalizing federation...", transient=True):
            if client.malicious:
                print(f"Client {client.index} is malicious, skipping finalization.")
                continue
            client.finalize()
        # self._compute_evaluation(self.rounds, client_to_eval)
        self.notify(event="finished", round=self.rounds + 1)


class FedHPIFCAClient_Adv(FedHPIFCAClient):
    
    def __init__(self, malicious_percentage=0, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.malicious = torch.rand(1).item() < malicious_percentage
        print(self.malicious)

    def fit(self, override_local_epochs = 0):
        if self.malicious:
            return super().fit(1)
        else:
            return super().fit(override_local_epochs)

    def send_model(self) -> None:
        if self.malicious:
            # self.channel.send(Message({i: torch.rand(self.prototypes[i,:].shape) for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
            print(f'client {self.index} is malicious, sending random prototypes')
            self.channel.send(Message(torch.rand(self.model.prototypes.data.shape),"prototypes", self.index, inmemory=True), "server")
        else:    
            self.channel.send(Message(copy.deepcopy(self.model.prototypes.data),"prototypes", self.index, inmemory=True), "server")


class FedHPIFCAServer_Adv(FedHPIFCAServer):

    def finalize(self):
        if self.rounds == 0:
            return
        client_to_eval = [client for client in self.clients if client.index in self._participants]
        self.broadcast_model(client_to_eval)
        for client in track(client_to_eval, "Finalizing federation...", transient=True):
            if client.malicious:
                print(f"Client {client.index} is malicious, skipping finalization.")
                continue
            client.finalize()
        # self._compute_evaluation(self.rounds, client_to_eval)
        self.notify(event="finished", round=self.rounds + 1)










class FedProto_Adv(PersonalizedFL):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.server.attach(self)

    def get_client_class(self) -> PFLClient:
        return FedProtoClient_Adv

    def get_server_class(self) -> type[Server]:
        return FedProtoServer
    

class FedHP_Adv(PersonalizedFL):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.server.attach(self)

    def get_client_class(self) -> PFLClient:
        return FedHPClient_Adv

    def get_server_class(self) -> type[Server]:
        return FedHPServer_Adv


class FedProtoIFCA_Adv(PersonalizedFL):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.server.attach(self)

    def get_client_class(self) -> PFLClient:
        return FedProtoIFCAClient_Adv

    def get_server_class(self) -> type[Server]:
        return FedProtoIFCAServer_Adv
    

class FedHPIFCA_Adv(PersonalizedFL):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.server.attach(self)

    def get_client_class(self) -> PFLClient:
        return FedHPIFCAClient_Adv

    def get_server_class(self) -> type[Server]:
        return FedHPIFCAServer_Adv