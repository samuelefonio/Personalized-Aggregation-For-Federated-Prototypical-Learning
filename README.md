# Repository for "Personalized Aggregation for Federated Prototypical Learning"
This is the official repository of "Personalized Aggregation for Federated Prototypical Learning", S. Fonio, B. Casella, M. Aldinucci, Proceedings of the 3^{rd} Workshop on Advancements on Federated Learning, European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases - ECML PKDD 2025, Porto, Portugal, 15-19 September 2025.  
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
