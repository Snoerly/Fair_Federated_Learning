## Adult (GIFAIR-FL)

python federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr sex
python federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr sex
python federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr sex  
python federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr sex  
python federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr sex
python federated_main.py --dataset adult --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr sex

## Bank (GIFAIR-FL)

python federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr age
python federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr age
python federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr age  
python federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr age  
python federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr age
python federated_main.py --dataset bank --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr age

## KDD Census (GIFAIR-FL)

python federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr ASEX
python federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr ASEX
python federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr ASEX  
python federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr ASEX  
python federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr ASEX
python federated_main.py --dataset census_income_kdd --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr ASEX

## Communities_crime (GIFAIR-FL)

python federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid feature-skew --sensitive_attr racepctblack
python federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid feature-skew --sensitive_attr racepctblack
python federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 0 --tabular_noniid label-skew --sensitive_attr racepctblack  
python federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 3 --iid 0 --tabular_noniid label-skew --sensitive_attr racepctblack  
python federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 10 --iid 1 --sensitive_attr racepctblack  
python federated_main.py --dataset communities_crime --model mlp --num_classes 2 --num_users 5 --iid 1 --sensitive_attr racepctblack

# Enforcing Group Fairness (neuer Algorithmus)


################################
# Adult
################################

# feature-skew
python main.py --dataset adult --num_users 5 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset adult --num_users 3 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1

# label-skew
python main.py --dataset adult --num_users 5 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset adult --num_users 3 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1

# iid
python main.py --dataset adult --num_users 10 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset adult --num_users 5 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1


################################
# Bank
################################

# feature-skew
python main.py --dataset bank --num_users 5 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset bank --num_users 3 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1

# label-skew
python main.py --dataset bank --num_users 5 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset bank --num_users 3 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1

# iid
python main.py --dataset bank --num_users 10 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset bank --num_users 5 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1


################################
# Census Income KDD
################################

# feature-skew
python main.py --dataset census --num_users 5 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset census --num_users 3 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1

# label-skew
python main.py --dataset census --num_users 5 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset census --num_users 3 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1

# iid
python main.py --dataset census --num_users 10 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset census --num_users 5 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1


################################
# Communities & Crime
################################

# feature-skew
python main.py --dataset communities --num_users 5 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset communities --num_users 3 --tabular_noniid feature-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1

# label-skew
python main.py --dataset communities --num_users 5 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset communities --num_users 3 --tabular_noniid label-skew --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1

# iid
python main.py --dataset communities --num_users 10 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1
python main.py --dataset communities --num_users 5 --iid --frac 0.1 --rounds 10 --local_epochs 10 --local_bs 10 --seed 1

FINISH!!
