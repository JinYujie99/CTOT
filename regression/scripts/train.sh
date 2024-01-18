python train.py  --dataset HousePrice --batch_size 128 --lr 2e-3 --wd 5e-4 \
                 --alpha 0.01 --lambda_value 1.0  --latent_size 64  --hidden_size 128  --context_size 16 --epochs 250 \
                 --kl_weight 0.1 --seed 42
