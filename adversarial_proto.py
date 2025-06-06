
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
from fedproto_C import FedProto_C_Client, FedProto_C_Server  # NOQA
from fedhp_C import FedHP_C_Client, FedHP_C_Server  # NOQA
from rich.progress import track
from fluke import FlukeENV

class FedProtoClient_Adv(FedProtoClient):
    
    def __init__(self, malicious_percentage=0, adv_factor=1, atk_type='random', *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.malicious = torch.rand(1).item() < malicious_percentage
        self.hyper_params.update(adv_factor=adv_factor, atk_type=atk_type)

    def send_model(self) -> None:
        if self.malicious:
            # print([self.global_protos[j] is not None for j in range(self.hyper_params.n_protos)])
            # self.channel.send(Message({i: torch.rand(self.prototypes[i].shape[0]) for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
            if all(self.global_protos[j] is not None for j in range(self.hyper_params.n_protos)):
                if self.hyper_params['atk_type'] == 'revert':
                    print(f'client {self.index} is malicious, sending global_prototypes reverted')
                    self.channel.send(Message({i: -self.hyper_params['adv_factor']*self.global_protos[i] for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
                else:
                    print(f'client {self.index} is malicious, sending random prototypes')
                    self.channel.send(Message({i: self.hyper_params['adv_factor']*torch.rand(self.prototypes[i].shape[0]) for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
            else:
                print(f'client {self.index} is malicious, sending random prototypes')
                self.channel.send(Message({i: self.hyper_params['adv_factor']*torch.rand(self.prototypes[i].shape[0]) for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
        else:    
            print(f'client {self.index} is good, sending the prototypes')
            self.channel.send(Message(self.prototypes, "model", self.index, inmemory=True), "server")

    def fit(self, override_local_epochs = 0):
        if self.malicious:
            if self.prototypes[0] == None:
                return super().fit(1)
        else:
            return super().fit(override_local_epochs)

    def evaluate(self, evaluator, test_set) -> dict[str, float]:
        if self.malicious:
            return {}
        else:
            return super().evaluate(evaluator, test_set)

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

class FedProto_C_Client_Adv(FedProto_C_Client):
    
    def __init__(self, malicious_percentage=0, adv_factor=1, atk_type='random', *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.malicious = torch.rand(1).item() < malicious_percentage
        self.hyper_params.update(adv_factor=adv_factor, atk_type=atk_type)

    def send_model(self) -> None:
        if self.malicious:
            if all(self.global_protos[j] is not None for j in range(self.hyper_params.n_protos)):
                if self.hyper_params['atk_type'] == 'revert':
                    print(f'client {self.index} is malicious, sending global_prototypes reverted')
                    self.channel.send(Message({i: -self.hyper_params['adv_factor']*self.global_protos[i] for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
                else:
                    print(f'client {self.index} is malicious, sending random prototypes')
                    self.channel.send(Message({i: self.hyper_params['adv_factor']*torch.rand(self.prototypes[i].shape[0]) for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
            else:
                print(f'client {self.index} is malicious, sending random prototypes')
                self.channel.send(Message({i: self.hyper_params['adv_factor']*torch.rand(self.prototypes[i].shape[0]) for i in range(self.hyper_params.n_protos)}, "model", self.index, inmemory=True), "server")
        else:    
            print(f'client {self.index} is good, sending the prototypes')
            self.channel.send(Message(self.prototypes, "model", self.index, inmemory=True), "server")

    def fit(self, override_local_epochs = 0):
        if self.malicious:
            if self.prototypes[0] == None:
                return super().fit(1)
        else:
            return super().fit(override_local_epochs)

    def evaluate(self, evaluator, test_set) -> dict[str, float]:
        if self.malicious:
            return {}
        else:
            return super().evaluate(evaluator, test_set)

class FedProto_C_Server_Adv(FedProto_C_Server):

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
    
    def __init__(self, malicious_percentage=0, adv_factor=1, atk_type='random', *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.malicious = torch.rand(1).item() < malicious_percentage
        self.hyper_params.update(adv_factor=adv_factor, atk_type=atk_type)

    def fit(self, override_local_epochs = 0):
        if self.malicious:
            if self.model.prototypes == None:
                return super().fit(1)
        else:
            return super().fit(override_local_epochs)

    def evaluate(self, evaluator, test_set) -> dict[str, float]:
        if self.malicious:
            return {}
        else:
            return super().evaluate(evaluator, test_set)

    def send_model(self) -> None:

        if self.malicious:
            if self.hyper_params['atk_type'] == 'revert':
                print(f'client {self.index} is malicious, sending global_prototypes reverted')
                self.channel.send(Message(-self.hyper_params['adv_factor']*copy.deepcopy(self.model.prototypes.data), "model", self.index, inmemory=True), "server")
            else:
                print(f'client {self.index} is malicious, sending random prototypes')
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


class FedHP_C_Client_Adv(FedHP_C_Client):
    
    def __init__(self, malicious_percentage=0, adv_factor=1, atk_type='random', *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.malicious = torch.rand(1).item() < malicious_percentage
        self.hyper_params.update(adv_factor=adv_factor, atk_type=atk_type)

    def fit(self, override_local_epochs = 0):
        if self.malicious:
            if self.model.prototypes == None:
                return super().fit(1)
        else:
            return super().fit(override_local_epochs)

    def evaluate(self, evaluator, test_set) -> dict[str, float]:
        if self.malicious:
            return {}
        else:
            return super().evaluate(evaluator, test_set)

    def send_model(self) -> None:
        if self.malicious:
            if self.hyper_params['atk_type'] == 'revert':
                print(f'client {self.index} is malicious, sending global_prototypes reverted')
                self.channel.send(Message(-self.hyper_params['adv_factor']*copy.deepcopy(self.model.prototypes.data), "model", self.index, inmemory=True), "server")
            else:
                print(f'client {self.index} is malicious, sending random prototypes')
                self.channel.send(Message(torch.rand(self.model.prototypes.data.shape),"prototypes", self.index, inmemory=True), "server")
        else:    
            self.channel.send(Message(copy.deepcopy(self.model.prototypes.data),"prototypes", self.index, inmemory=True), "server")


class FedHP_C_Server_Adv(FedHP_C_Server):

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
        return FedProtoServer_Adv
    

class FedHP_Adv(PersonalizedFL):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.server.attach(self)

    def get_client_class(self) -> PFLClient:
        return FedHPClient_Adv

    def get_server_class(self) -> type[Server]:
        return FedHPServer_Adv


class FedProto_C_Adv(PersonalizedFL):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.server.attach(self)

    def get_client_class(self) -> PFLClient:
        return FedProto_C_Client_Adv

    def get_server_class(self) -> type[Server]:
        return FedProto_C_Server_Adv
    

class FedHP_C_Adv(PersonalizedFL):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.server.attach(self)

    def get_client_class(self) -> PFLClient:
        return FedHP_C_Client_Adv

    def get_server_class(self) -> type[Server]:
        return FedHP_C_Server_Adv