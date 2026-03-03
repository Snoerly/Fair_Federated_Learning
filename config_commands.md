# GIFAIR-FL

## Adult

### feature-skew

python code/GIFAIR-FL/federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr sex  
python code/GIFAIR-FL/federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr sex  

### label-skew

python code/GIFAIR-FL/federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr sex  
python code/GIFAIR-FL/federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr sex  

### iid

python code/GIFAIR-FL/federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr sex  
python code/GIFAIR-FL/federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr sex  

## Bank

### feature-skew

python code/GIFAIR-FL/federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr age  
python code/GIFAIR-FL/federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr age  

### label-skew

python code/GIFAIR-FL/federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr age  
python code/GIFAIR-FL/federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr age  

### iid

python code/GIFAIR-FL/federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr age  
python code/GIFAIR-FL/federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr age  

## KDD Census

### feature-skew

python code/GIFAIR-FL/federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr ASEX  
python code/GIFAIR-FL/federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr ASEX  

### label-skew

python code/GIFAIR-FL/federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr ASEX  
python code/GIFAIR-FL/federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr ASEX  

### iid

python code/GIFAIR-FL/federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr ASEX  
python code/GIFAIR-FL/federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr ASEX  

## Communities_crime

### feature-skew

python code/GIFAIR-FL/federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr racepctblack  
python code/GIFAIR-FL/federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr racepctblack  

### label-skew

python code/GIFAIR-FL/federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr racepctblack  
python code/GIFAIR-FL/federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr racepctblack  

### iid

python code/GIFAIR-FL/federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr racepctblack  
python code/GIFAIR-FL/federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr racepctblack  

# Enforcing Group Fairness

## Adult

### feature-skew

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset adult --num_users 5 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset adult --num_users 3 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

### label-skew

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset adult --num_users 5 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset adult --num_users 3 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

### iid

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset adult --num_users 10 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset adult --num_users 5 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

## Bank

### feature-skew

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset bank --num_users 5 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset bank --num_users 3 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

### label-skew

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset bank --num_users 5 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset bank --num_users 3 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

### iid

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset bank --num_users 10 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset bank --num_users 5 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

## Census Income KDD

### feature-skew

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset census --num_users 5 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset census --num_users 3 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

### label-skew

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset census --num_users 5 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset census --num_users 3 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

### iid

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset census --num_users 10 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset census --num_users 5 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

## Communities & Crime

### feature-skew

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset communities --num_users 5 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset communities --num_users 3 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

### label-skew

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset communities --num_users 5 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset communities --num_users 3 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  

### iid

python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset communities --num_users 10 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1  
python code/Enforcing_Group_Fairness_in_Privacy_Preserving/main.py --dataset communities --num_users 5 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1   


# FairTrade

## Adult

### feature-skew

python code/FairTrade/fairtrade_federated.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr sex --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset adult --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr sex --fairness_notion stat_parity --fairness_lambda 1.0  

### label-skew

python code/FairTrade/fairtrade_federated.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr sex --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset adult --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr sex --fairness_notion stat_parity --fairness_lambda 1.0  

### iid

python code/FairTrade/fairtrade_federated.py --dataset adult --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr sex --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr sex --fairness_notion stat_parity --fairness_lambda 1.0  

## Bank

### feature-skew

python code/FairTrade/fairtrade_federated.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr age --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset bank --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr age --fairness_notion stat_parity --fairness_lambda 1.0  

### label-skew

python code/FairTrade/fairtrade_federated.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr age --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset bank --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr age --fairness_notion stat_parity --fairness_lambda 1.0  

### iid

python code/FairTrade/fairtrade_federated.py --dataset bank --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr age --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr age --fairness_notion stat_parity --fairness_lambda 1.0  

## KDD Census

### feature-skew
 
python code/FairTrade/fairtrade_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr ASEX --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr ASEX --fairness_notion stat_parity --fairness_lambda 1.0  

### label-skew

python code/FairTrade/fairtrade_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr ASEX --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr ASEX --fairness_notion stat_parity --fairness_lambda 1.0  

### iid

python code/FairTrade/fairtrade_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr ASEX --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr ASEX --fairness_notion stat_parity --fairness_lambda 1.0  

## Communities_crime

### feature-skew

python code/FairTrade/fairtrade_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr racepctblack --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr racepctblack --fairness_notion stat_parity --fairness_lambda 1.0  

### label-skew

python code/FairTrade/fairtrade_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr racepctblack --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr racepctblack --fairness_notion stat_parity --fairness_lambda 1.0  

### iid

python code/FairTrade/fairtrade_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr racepctblack --fairness_notion stat_parity --fairness_lambda 1.0  
python code/FairTrade/fairtrade_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr racepctblack --fairness_notion stat_parity --fairness_lambda 1.0  


# FedAvg

## Adult

### feature-skew

python code/FedAvg/fedavg_federated.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr sex  
python code/FedAvg/fedavg_federated.py --dataset adult --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr sex  
 
### label-skew

python code/FedAvg/fedavg_federated.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr sex  
python code/FedAvg/fedavg_federated.py --dataset adult --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr sex  

### iid

python code/FedAvg/fedavg_federated.py --dataset adult --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr sex  
python code/FedAvg/fedavg_federated.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr sex  

## Bank

### feature-skew

python code/FedAvg/fedavg_federated.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr age  
python code/FedAvg/fedavg_federated.py --dataset bank --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr age  

### label-skew

python code/FedAvg/fedavg_federated.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr age  
python code/FedAvg/fedavg_federated.py --dataset bank --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr age  

### iid

python code/FedAvg/fedavg_federated.py --dataset bank --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr age  
python code/FedAvg/fedavg_federated.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr age  

## KDD Census

### feature-skew

python code/FedAvg/fedavg_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr ASEX  
python code/FedAvg/fedavg_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr ASEX  

### label-skew

python code/FedAvg/fedavg_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr ASEX  
python code/FedAvg/fedavg_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr ASEX  

### iid

python code/FedAvg/fedavg_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr ASEX  
python code/FedAvg/fedavg_federated.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr ASEX  

## Communities_crime

### feature-skew

python code/FedAvg/fedavg_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr racepctblack  
python code/FedAvg/fedavg_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr racepctblack  
 
### label-skew
 
python code/FedAvg/fedavg_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr racepctblack  
python code/FedAvg/fedavg_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr racepctblack  
 
### iid

python code/FedAvg/fedavg_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr racepctblack  
python code/FedAvg/fedavg_federated.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr racepctblack  

