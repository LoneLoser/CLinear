idx=$1
random_seed=$2
gpu=$3
seq_len=(144 144 144 144 144 216)
label_len=(48 48 48 72 120 192)
pred_len=(1 12 24 48 96 168)
lr=(0.003 0.003 0.003 0.003 0.0003 0.0001)

python run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --model CLinear \
  \
  --data custom \
  --root_path ./data/site/ \
  --data_path temperature.csv \
  --freq h \
  --seq_len ${seq_len[$idx]} \
  --label_len ${label_len[$idx]} \
  --pred_len ${pred_len[$idx]} \
  --period 24 \
  \
  --covar_layers 0 \
  --enc_in 10 \
  --dropout 0.05 \
  --conv_activation none \
  --fc_activation relu \
  \
  --batch_size 128 \
  --learning_rate ${lr[$idx]} \
  --inverse 1 \
  --gpu $gpu
