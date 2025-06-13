# Repository for "Personalized Aggregation for Federated Prototypical Learning"
This is the official repository of "Personalized Aggregation for Federated Prototypical Learning".  
The algorithm is implemented in fluke and compare with baselines implemented in it.  
To reproduce the experiments, install a venv:

```
python -m venv venv  
source venv/bin/activate
pip install fluke-fl
```

Run the algorithms through the command:
```
fluke sweep config/exp.yaml adv_fedproto_C.yaml
```