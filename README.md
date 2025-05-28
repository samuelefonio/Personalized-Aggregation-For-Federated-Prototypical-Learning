# Repository for "Clustered Federated Leanring with Prototypes"
This is the official repository of "Clustered Federated Learning with Prototypes".  
The algorithm is implemented in fluke and compare with baselines implemented in it.  
To reproduce the experiments, install a venv:

```
python -m venv venv  
source venv/bin/activate
pip install fluke-fl
```

Run the algorithms through the command:
```
fluke federation config/exp.yaml fedprotoIFCA.yaml
```