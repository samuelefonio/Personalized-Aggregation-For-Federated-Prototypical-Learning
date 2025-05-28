
import json
import logging
import os
import sys
import time
from typing import Any, Collection, Literal, Union

import clearml
import numpy as np
from pandas import DataFrame
from psutil import Process
from rich import print as rich_print
from rich.logging import RichHandler
from rich.panel import Panel
from rich.pretty import Pretty
from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter

import wandb

sys.path.append(".")
sys.path.append("..")

from fluke import DDict  # NOQA
from fluke.comm import ChannelObserver, Message  # NOQA
from fluke.utils import bytes2human, get_class_from_qualified_name  # NOQA
from fluke.utils import ClientObserver, ServerObserver, get_class_from_str  # NOQA

__all__ = ["Log", "DebugLog", "TensorboardLog", "WandBLog", "ClearMLLog", "get_logger"]


class Log(ServerObserver, ChannelObserver, ClientObserver):
    """Basic logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process. The logging happens in the console.

    Attributes:
        global_eval (dict): The global evaluation metrics.
        locals_eval (dict): The clients local model evaluation metrics on the server's test set.
        prefit_eval (dict): The clients' pre-fit evaluation metrics.
        postfit_eval (dict): The clients' post-fit evaluation metrics.
        locals_eval_summary (dict): The mean of the clients local model evaluation metrics.
        prefit_eval_summary (dict): The mean of the clients pre-fit evaluation metrics.
        postfit_eval_summary (dict): The mean of the clients post-fit evaluation metrics.
        comm_costs (dict): The communication costs.
        current_round (int): The current round.
    """

    def __init__(self, **kwargs):
        self.global_eval: dict = {}  # round -> evals
        self.locals_eval: dict = {}  # round -> {client_id -> evals}
        self.prefit_eval: dict = {}  # round -> {client_id -> evals}
        self.postfit_eval: dict = {}  # round -> {client_id -> evals}
        self.locals_eval_summary: dict = {}  # round -> evals (mean across clients)
        self.prefit_eval_summary: dict = {}  # round -> evals (mean across clients)
        self.postfit_eval_summary: dict = {}  # round -> evals (mean across clients)
        self.comm_costs: dict = {0: 0}  # round -> comm_cost
        self.mem_costs: dict = {}
        self.current_round: int = 0
        self.custom_fields: dict = {}

    def log(self, message: str) -> None:
        """Log a message.

        Args:
            message (str): The message to log.
        """
        rich_print(message)

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        """Add a scalar to the logger.

        Args:
            key (Any): The key of the scalar.
            value (float): The value of the scalar.
            round (int): The round.
        """
        if round not in self.custom_fields:
            self.custom_fields[round] = {}
        self.custom_fields[round][key] = value

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        """Add scalars to the logger.

        Args:
            key (Any): The main key of the scalars.
            values (dict[str, float]): The key-value pairs of the scalars.
            round (int): The round.
        """
        if round not in self.custom_fields:
            self.custom_fields[round] = {}
        for k, v in values.items():
            self.custom_fields[round][f"{key}/{k}"] = v

    def pretty_log(self, data: Any, title: str) -> None:
        """Log a pretty-printed data.

        Args:
            data (Any): The data to log.
            title (str): The title of the data.
        """
        rich_print(Panel(Pretty(data, expand_all=True), title=title, width=100))

    def init(self, **kwargs) -> None:
        """Initialize the logger.
        The initialization is done by printing the configuration in the console.

        Args:
            **kwargs: The configuration.
        """
        if kwargs:
            rich_print(Panel(Pretty(kwargs, expand_all=True), title="Configuration", width=100))

    def start_round(self, round: int, global_model: Module) -> None:
        self.comm_costs[round] = 0
        self.current_round = round

        if round == 1 and self.comm_costs[0] > 0:
            rich_print(
                Panel(
                    Pretty({"comm_costs": self.comm_costs[0]}),
                    title=f"Round: {round - 1}",
                    width=100,
                )
            )

    def end_round(self, round: int) -> None:
        stats = {}
        # Pre-fit summary
        if self.prefit_eval and round in self.prefit_eval and self.prefit_eval[round]:
            client_mean = (
                DataFrame(self.prefit_eval[round].values()).mean(numeric_only=True).to_dict()
            )
            client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
            self.prefit_eval_summary[round] = client_mean
            stats["pre-fit"] = client_mean

        # Post-fit summary
        if self.postfit_eval and round in self.postfit_eval and self.postfit_eval[round]:
            client_mean = (
                DataFrame(self.postfit_eval[round].values()).mean(numeric_only=True).to_dict()
            )
            client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
            self.postfit_eval_summary[round] = client_mean
            stats["post-fit"] = client_mean

        # Locals summary
        if self.locals_eval and round in self.locals_eval and self.locals_eval[round]:
            client_mean = (
                DataFrame(list(self.locals_eval[round].values())).mean(numeric_only=True).to_dict()
            )
            client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
            self.locals_eval_summary[round] = client_mean
            stats["locals"] = self.locals_eval_summary[round]

        # Global summary
        if self.global_eval and round in self.global_eval and self.global_eval[round]:
            stats["global"] = self.global_eval[round]

        stats["comm_cost"] = self.comm_costs[round]
        proc = Process(os.getpid())
        self.mem_costs[round] = proc.memory_full_info().uss

        if self.custom_fields and round in self.custom_fields:
            stats.update(self.custom_fields[round])

        rich_print(Panel(Pretty(stats, expand_all=True), title=f"Round: {round}", width=100))
        rich_print(
            f"  Memory usage: {bytes2human(self.mem_costs[round])}"
            + f"[{proc.memory_percent():.2f} %]"
        )

    def client_evaluation(
        self,
        round: int,
        client_id: int,
        phase: Literal["pre-fit", "post-fit"],
        evals: dict[str, float],
        **kwargs,
    ) -> None:
        if round == -1:
            round = self.current_round + 1
        dict_ref = self.prefit_eval if phase == "pre-fit" else self.postfit_eval
        if round not in dict_ref:
            dict_ref[round] = {}
        dict_ref[round][client_id] = evals

    def server_evaluation(
        self,
        round: int,
        eval_type: Literal["global", "locals"],
        evals: Union[dict[str, float], dict[int, dict[str, float]]],
        **kwargs,
    ) -> None:

        if eval_type == "global" and evals:
            self.global_eval[round] = evals
        elif eval_type == "locals" and evals:
            self.locals_eval[round] = evals

    def message_received(self, by: Any, message: Message) -> None:
        """Update the communication costs.

        Args:
            by (Any): The sender of the message.
            message (Message): The message received.
        """
        self.comm_costs[self.current_round] += message.size

    def finished(self, round: int) -> None:
        stats = {}

        # Pre-fit summary
        if self.prefit_eval:

            if round in self.prefit_eval and self.prefit_eval[round]:
                last_round = round
            else:
                last_round = max(self.prefit_eval.keys())
            if last_round not in self.prefit_eval_summary:
                client_mean = (
                    DataFrame(self.prefit_eval[last_round].values())
                    .mean(numeric_only=True)
                    .to_dict()
                )
                client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
                self.prefit_eval_summary[last_round] = client_mean

            stats["pre-fit"] = self.prefit_eval_summary[last_round]
            stats["pre-fit"]["round"] = last_round

        # Locals summary
        if self.locals_eval:
            last_round = max(self.locals_eval.keys())
            stats["locals"] = self.locals_eval_summary[last_round]
            stats["locals"]["round"] = last_round

        # Post-fit summary
        if self.postfit_eval:
            if round in self.postfit_eval and self.postfit_eval[round]:
                last_round = round
            else:
                last_round = max(self.postfit_eval.keys())

            if last_round not in self.postfit_eval_summary:
                client_mean = (
                    DataFrame(self.postfit_eval[last_round].values())
                    .mean(numeric_only=True)
                    .to_dict()
                )
                client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
                self.postfit_eval_summary[last_round] = client_mean
            stats["post-fit"] = self.postfit_eval_summary[last_round]
            stats["post-fit"]["round"] = last_round

        # Global summary
        if self.global_eval:
            last_round = max(self.global_eval.keys())
            stats["global"] = self.global_eval[last_round]
            stats["global"]["round"] = last_round

        if stats:
            rich_print(
                Panel(
                    Pretty(stats, expand_all=True),
                    title="Overall Performance",
                    width=100,
                )
            )

        rich_print(
            Panel(
                Pretty({"comm_costs": sum(self.comm_costs.values())}, expand_all=True),
                title="Total communication cost",
                width=100,
            )
        )

    def interrupted(self) -> None:
        rich_print("\n[bold italic yellow]The experiment has been interrupted by the user.")

    def early_stop(self, round: int) -> None:
        return self.end_round(round)

    def track_item(self, round: int, item: str, value: Any) -> None:
        self.add_scalar(item, value, round)

    def save(self, path: str) -> None:
        """Save the logger's history to a JSON file.

        Args:
            path (str): The path to the JSON file.
        """
        json_to_save = {
            "perf_global": self.global_eval,
            "comm_costs": self.comm_costs,
            "perf_locals": self.locals_eval_summary,
            "perf_prefit": self.prefit_eval_summary,
            "perf_postfit": self.postfit_eval_summary,
            "mem_costs": self.mem_costs,
            "custom_fields": self.custom_fields,
        }
        with open(path, "w") as f:
            json.dump(json_to_save, f, indent=4)

    def close(self) -> None:
        """Close the logger."""
        pass

class MyWandBLog(Log):
    """Weights and Biases logger.
    This class is used to log the performance of the global model and the communication costs during
    the federated learning process on Weights and Biases.

    See Also:
        For more information on Weights and Biases, see the `Weights and Biases documentation
        <https://docs.wandb.ai/>`_.

    Args:
        **config: The configuration for Weights and Biases.
    """

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self.run = None

    def init(self, **kwargs) -> None:
        super().init(**kwargs)
        self.config["config"] = kwargs
        self.run = wandb.init(**self.config)

    def add_scalar(self, key: Any, value: float, round: int) -> None:
        super().add_scalar(key, value, round)
        return self.run.log({key: value}, step=round)

    def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
        super().add_scalars(key, values, round)
        return self.run.log({f"{key}/{k}": v for k, v in values.items()}, step=round)

    def start_round(self, round: int, global_model: Module) -> None:
        super().start_round(round, global_model)
        if round == 1 and self.comm_costs[0] > 0:
            self.run.log({"comm_costs": self.comm_costs[0]})

    def end_round(self, round: int) -> None:
        
        super().end_round(round)
        if round in self.global_eval:
            self.run.log({"global": self.global_eval[round]}, step=round)
        self.run.log({"comm_cost": self.comm_costs[round]}, step=round)

        if self.prefit_eval_summary and round in self.prefit_eval_summary:
            self.run.log({"prefit": self.prefit_eval_summary[round]}, step=round)

        if self.postfit_eval_summary and round in self.postfit_eval_summary:
            # if self.postfit_eval_summary and round in self.postfit_eval_summary:
            # print(self.postfit_eval[round].values())
            # client_mean = DataFrame(self.postfit_eval[round].values()).mean(
            #     numeric_only=True).to_dict()
            # print(client_mean)
            client_std = DataFrame(self.postfit_eval[round].values()).std(
                numeric_only=True).to_dict()
            
            client_std = {k: float(np.round(float(v), 5)) for k, v in client_std.items()}
            client_std = {f"{k}_std": v for k, v in client_std.items()}
            # print(client_std)
            rich_print(Panel(Pretty(client_std, expand_all=True), title='standard_deviations', width=100))
            
            self.run.log(client_std, step=round)
            self.run.log({"postfit": self.postfit_eval_summary[round]}, step=round)

        
        if self.postfit_eval_summary and round in self.postfit_eval_summary:
            self.run.log({"postfit": self.postfit_eval_summary[round]}, step=round)

        if self.locals_eval_summary and round in self.locals_eval_summary:
            self.run.log({"locals": self.locals_eval_summary[round]}, step=round)

    def finished(self, round: int) -> None:
        super().finished(round)

        if self.prefit_eval_summary and round in self.prefit_eval_summary:
            self.run.log({"prefit": self.prefit_eval_summary[round]}, step=round)

        if self.postfit_eval_summary and round in self.postfit_eval_summary:
            client_std = DataFrame(self.postfit_eval[round].values()).std(
                numeric_only=True).to_dict()
            print(len(self.postfit_eval[round].values()))
            client_std = {k: float(np.round(float(v), 5)) for k, v in client_std.items()}
            client_std = {f"{k}_std": v for k, v in client_std.items()}
            # print(client_std)
            rich_print(Panel(Pretty(client_std, expand_all=True), title='standard_deviations', width=100))
            
            self.run.log(client_std, step=round)
            self.run.log({"postfit": self.postfit_eval_summary[round]}, step=round)

        if self.locals_eval_summary and round in self.locals_eval_summary:
            self.run.log({"locals": self.locals_eval_summary[round]}, step=round)

    def close(self) -> None:
        self.run.finish()

    def save(self, path: str) -> None:
        super().save(path)


def get_logger(lname: str, **kwargs) -> Log:
    """Get a logger from its name.
    This function is used to get a logger from its name. It is used to dynamically import loggers.
    The supported loggers are the ones defined in the ``fluke.utils.log`` module, but it can handle
    any logger defined by the user.

    Note:
        To use a custom logger, it must be defined in a module and the full model name must be
        provided in the configuration file. For example, if the logger ``MyLogger`` is defined in
        the module ``my_module`` (i.e., a file called ``my_module.py``), the logger name must be
        ``my_module.MyLogger``. The other logger's parameters must be passed as in the following
        example:

        .. code-block:: yaml

            logger:
                name: my_module.MyLogger
                param1: value1
                param2: value2
                ...


    Args:
        lname (str): The name of the logger.
        **kwargs: The keyword arguments to pass to the logger's constructor.

    Returns:
        Log | DebugLog | WandBLog | ClearMLLog | TensorboardLog: The logger.
    """
    if "." in lname and not lname.startswith("fluke.utils.log"):
        return get_class_from_qualified_name(lname)(**kwargs)
    return get_class_from_str("fluke.utils.log", lname)(**kwargs)

# import json
# import logging
# import os
# import sys
# import time
# from typing import Any, Collection, Literal, Union

# import clearml
# import numpy as np
# from pandas import DataFrame
# from psutil import Process
# from rich import print as rich_print
# from rich.logging import RichHandler
# from rich.panel import Panel
# from rich.pretty import Pretty
# from torch.nn import Module
# from torch.utils.tensorboard import SummaryWriter

# import wandb

# sys.path.append(".")
# sys.path.append("..")

# from fluke import DDict  # NOQA
# from fluke.comm import ChannelObserver, Message  # NOQA
# from fluke.utils import bytes2human, get_class_from_qualified_name  # NOQA
# from fluke.utils import ClientObserver, ServerObserver, get_class_from_str  # NOQA

# __all__ = ["Log", "DebugLog", "TensorboardLog", "WandBLog", "ClearMLLog", "get_logger"]



# class Log(ServerObserver, ChannelObserver, ClientObserver):
#     """Basic logger.
#     This class is used to log the performance of the global model and the communication costs during
#     the federated learning process. The logging happens in the console.

#     Attributes:
#         global_eval (dict): The global evaluation metrics.
#         locals_eval (dict): The clients local model evaluation metrics on the server's test set.
#         prefit_eval (dict): The clients' pre-fit evaluation metrics.
#         postfit_eval (dict): The clients' post-fit evaluation metrics.
#         locals_eval_summary (dict): The mean of the clients local model evaluation metrics.
#         prefit_eval_summary (dict): The mean of the clients pre-fit evaluation metrics.
#         postfit_eval_summary (dict): The mean of the clients post-fit evaluation metrics.
#         comm_costs (dict): The communication costs.
#         current_round (int): The current round.
#     """

#     def __init__(self, **kwargs):
#         self.global_eval: dict = {}  # round -> evals
#         self.locals_eval: dict = {}  # round -> {client_id -> evals}
#         self.prefit_eval: dict = {}  # round -> {client_id -> evals}
#         self.postfit_eval: dict = {}  # round -> {client_id -> evals}
#         self.locals_eval_summary: dict = {}  # round -> evals (mean across clients)
#         self.prefit_eval_summary: dict = {}  # round -> evals (mean across clients)
#         self.postfit_eval_summary: dict = {}  # round -> evals (mean across clients)
#         self.comm_costs: dict = {0: 0}  # round -> comm_cost
#         self.mem_costs: dict = {}
#         self.current_round: int = 0
#         self.custom_fields: dict = {}

#     def log(self, message: str) -> None:
#         """Log a message.

#         Args:
#             message (str): The message to log.
#         """
#         rich_print(message)

#     def add_scalar(self, key: Any, value: float, round: int) -> None:
#         """Add a scalar to the logger.

#         Args:
#             key (Any): The key of the scalar.
#             value (float): The value of the scalar.
#             round (int): The round.
#         """
#         if round not in self.custom_fields:
#             self.custom_fields[round] = {}
#         self.custom_fields[round][key] = value

#     def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
#         """Add scalars to the logger.

#         Args:
#             key (Any): The main key of the scalars.
#             values (dict[str, float]): The key-value pairs of the scalars.
#             round (int): The round.
#         """
#         if round not in self.custom_fields:
#             self.custom_fields[round] = {}
#         for k, v in values.items():
#             self.custom_fields[round][f"{key}/{k}"] = v

#     def pretty_log(self, data: Any, title: str) -> None:
#         """Log a pretty-printed data.

#         Args:
#             data (Any): The data to log.
#             title (str): The title of the data.
#         """
#         rich_print(Panel(Pretty(data, expand_all=True), title=title, width=100))

#     def init(self, **kwargs) -> None:
#         """Initialize the logger.
#         The initialization is done by printing the configuration in the console.

#         Args:
#             **kwargs: The configuration.
#         """
#         if kwargs:
#             rich_print(Panel(Pretty(kwargs, expand_all=True), title="Configuration", width=100))

#     def start_round(self, round: int, global_model: Module) -> None:
#         self.comm_costs[round] = 0
#         self.current_round = round

#         if round == 1 and self.comm_costs[0] > 0:
#             rich_print(
#                 Panel(
#                     Pretty({"comm_costs": self.comm_costs[0]}),
#                     title=f"Round: {round - 1}",
#                     width=100,
#                 )
#             )

#     def end_round(self, round: int) -> None:
#         stats = {}
#         # Pre-fit summary
#         if self.prefit_eval and round in self.prefit_eval and self.prefit_eval[round]:
#             client_mean = (
#                 DataFrame(self.prefit_eval[round].values()).mean(numeric_only=True).to_dict()
#             )
#             client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
#             self.prefit_eval_summary[round] = client_mean
#             stats["pre-fit"] = client_mean

#         # Post-fit summary
#         if self.postfit_eval and round in self.postfit_eval and self.postfit_eval[round]:
#             client_mean = (
#                 DataFrame(self.postfit_eval[round].values()).mean(numeric_only=True).to_dict()
#             )
#             client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
#             self.postfit_eval_summary[round] = client_mean
#             stats["post-fit"] = client_mean

#         # Locals summary
#         if self.locals_eval and round in self.locals_eval and self.locals_eval[round]:
#             client_mean = (
#                 DataFrame(list(self.locals_eval[round].values())).mean(numeric_only=True).to_dict()
#             )
#             client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
#             self.locals_eval_summary[round] = client_mean
#             stats["locals"] = self.locals_eval_summary[round]

#         # Global summary
#         if self.global_eval and round in self.global_eval and self.global_eval[round]:
#             stats["global"] = self.global_eval[round]

#         stats["comm_cost"] = self.comm_costs[round]
#         proc = Process(os.getpid())
#         self.mem_costs[round] = proc.memory_full_info().uss

#         if self.custom_fields and round in self.custom_fields:
#             stats.update(self.custom_fields[round])

#         rich_print(Panel(Pretty(stats, expand_all=True), title=f"Round: {round}", width=100))
#         rich_print(
#             f"  Memory usage: {bytes2human(self.mem_costs[round])}"
#             + f"[{proc.memory_percent():.2f} %]"
#         )

#     def client_evaluation(
#         self,
#         round: int,
#         client_id: int,
#         phase: Literal["pre-fit", "post-fit"],
#         evals: dict[str, float],
#         **kwargs,
#     ) -> None:

#         if round == -1:
#             round = self.current_round + 1
#         dict_ref = self.prefit_eval if phase == "pre-fit" else self.postfit_eval
#         if round not in dict_ref:
#             dict_ref[round] = {}
#         dict_ref[round][client_id] = evals

#     def server_evaluation(
#         self,
#         round: int,
#         eval_type: Literal["global", "locals"],
#         evals: Union[dict[str, float], dict[int, dict[str, float]]],
#         **kwargs,
#     ) -> None:

#         if eval_type == "global" and evals:
#             self.global_eval[round] = evals
#         elif eval_type == "locals" and evals:
#             self.locals_eval[round] = evals

#     def message_received(self, by: Any, message: Message) -> None:
#         """Update the communication costs.

#         Args:
#             by (Any): The sender of the message.
#             message (Message): The message received.
#         """
#         self.comm_costs[self.current_round] += message.size

#     def finished(self, round: int) -> None:
#         stats = {}

#         # Pre-fit summary
#         if self.prefit_eval and round in self.prefit_eval and self.prefit_eval[round]:
#             client_mean = (
#                 DataFrame(self.prefit_eval[round].values()).mean(numeric_only=True).to_dict()
#             )
#             client_mean = {k: float(np.round(float(v), 5)) for k, v in client_mean.items()}
#             self.prefit_eval_summary[round] = client_mean
#             stats["pre-fit"] = client_mean

#         # Locals summary
#         if self.locals_eval:
#             stats["locals"] = self.locals_eval_summary[max(self.locals_eval.keys())]

#         # Post-fit summary
#         if self.postfit_eval:
#             stats["post-fit"] = self.postfit_eval_summary[max(self.postfit_eval.keys())]

#         # Global summary
#         if self.global_eval:
#             stats["global"] = self.global_eval[max(self.global_eval.keys())]

#         if stats:
#             rich_print(
#                 Panel(
#                     Pretty(stats, expand_all=True),
#                     title="Overall Performance",
#                     width=100,
#                 )
#             )

#         rich_print(
#             Panel(
#                 Pretty({"comm_costs": sum(self.comm_costs.values())}, expand_all=True),
#                 title="Total communication cost",
#                 width=100,
#             )
#         )

#     def interrupted(self) -> None:
#         rich_print("\n[bold italic yellow]The experiment has been interrupted by the user.")

#     def early_stop(self, round: int) -> None:
#         return self.end_round(round)

#     def track_item(self, round: int, item: str, value: Any) -> None:
#         self.add_scalar(item, value, round)

#     def save(self, path: str) -> None:
#         """Save the logger's history to a JSON file.

#         Args:
#             path (str): The path to the JSON file.
#         """
#         json_to_save = {
#             "perf_global": self.global_eval,
#             "comm_costs": self.comm_costs,
#             "perf_locals": self.locals_eval_summary,
#             "perf_prefit": self.prefit_eval_summary,
#             "perf_postfit": self.postfit_eval_summary,
#             "mem_costs": self.mem_costs,
#             "custom_fields": self.custom_fields,
#         }
#         with open(path, "w") as f:
#             json.dump(json_to_save, f, indent=4)

#     def close(self) -> None:
#         """Close the logger."""
#         pass


# class MyWandBLog(Log):
#     """Weights and Biases logger.
#     This class is used to log the performance of the global model and the communication costs during
#     the federated learning process on Weights and Biases.

#     See Also:
#         For more information on Weights and Biases, see the `Weights and Biases documentation
#         <https://docs.wandb.ai/>`_.

#     Args:
#         **config: The configuration for Weights and Biases.
#     """

#     def __init__(self, **config):
#         super().__init__(**config)
#         self.config = config
#         self.run = None

#     def init(self, **kwargs) -> None:
#         super().init(**kwargs)
#         self.config["config"] = kwargs
#         self.run = wandb.init(**self.config)

#     def add_scalar(self, key: Any, value: float, round: int) -> None:
#         super().add_scalar(key, value, round)
#         return self.run.log({key: value}, step=round)

#     def add_scalars(self, key: Any, values: dict[str, float], round: int) -> None:
#         super().add_scalars(key, values, round)
#         return self.run.log({f"{key}/{k}": v for k, v in values.items()}, step=round)

#     def start_round(self, round: int, global_model: Module) -> None:
#         super().start_round(round, global_model)
#         if round == 1 and self.comm_costs[0] > 0:
#             self.run.log({"comm_costs": self.comm_costs[0]})

#     def end_round(self, round: int) -> None:
#         super().end_round(round)

#         if self.postfit_eval_summary and round in self.postfit_eval_summary:
#             # if self.postfit_eval_summary and round in self.postfit_eval_summary:
#             # print(self.postfit_eval[round].values())
#             # client_mean = DataFrame(self.postfit_eval[round].values()).mean(
#             #     numeric_only=True).to_dict()
#             # print(client_mean)
#             client_std = DataFrame(self.postfit_eval[round].values()).std(
#                 numeric_only=True).to_dict()
            
#             client_std = {k: float(np.round(float(v), 5)) for k, v in client_std.items()}
#             client_std = {f"{k}_std": v for k, v in client_std.items()}
#             # print(client_std)
#             rich_print(Panel(Pretty(client_std, expand_all=True), title='standard_deviations', width=100))
            
#             self.run.log(client_std, step=round)
#             # self.run.log({"postfit": self.postfit_eval_summary[round]}, step=round)

#     def finished(self, round: int) -> None:
#         super().finished(round)

#         if self.prefit_eval_summary and round in self.prefit_eval_summary:
#             self.run.log({"prefit": self.prefit_eval_summary[round]}, step=round)

#         if self.locals_eval_summary and round in self.locals_eval_summary:
#             self.run.log({"locals": self.locals_eval_summary[round]}, step=round)

#     def close(self) -> None:
#         self.run.finish()

#     def save(self, path: str) -> None:
#         super().save(path)



# def get_logger(lname: str, **kwargs) -> Log:
#     """Get a logger from its name.
#     This function is used to get a logger from its name. It is used to dynamically import loggers.
#     The supported loggers are the ones defined in the ``fluke.utils.log`` module, but it can handle
#     any logger defined by the user.

#     Note:
#         To use a custom logger, it must be defined in a module and the full model name must be
#         provided in the configuration file. For example, if the logger ``MyLogger`` is defined in
#         the module ``my_module`` (i.e., a file called ``my_module.py``), the logger name must be
#         ``my_module.MyLogger``. The other logger's parameters must be passed as in the following
#         example:

#         .. code-block:: yaml

#             logger:
#                 name: my_module.MyLogger
#                 param1: value1
#                 param2: value2
#                 ...


#     Args:
#         lname (str): The name of the logger.
#         **kwargs: The keyword arguments to pass to the logger's constructor.

#     Returns:
#         Log | DebugLog | WandBLog | ClearMLLog | TensorboardLog: The logger.
#     """
#     if "." in lname and not lname.startswith("fluke.utils.log"):
#         return get_class_from_qualified_name(lname)(**kwargs)
#     return get_class_from_str("fluke.utils.log", lname)(**kwargs)