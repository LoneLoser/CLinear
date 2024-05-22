idx=$1
random_seed=$2
gpu=$3
seq_len=(864 864 1008 1008)
label_len=(96 192 336 720)
pred_len=(96 192 336 720)

python run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --model CLinear \
  \
  --data custom \
  --root_path ./data/weather/ \
  --data_path weather.csv \
  --freq 10min \
  --seq_len ${seq_len[$idx]} \
  --label_len ${label_len[$idx]} \
  --pred_len ${pred_len[$idx]} \
  --period 144 \
  \
  --covar_layers 0 \
  --enc_in 21 \
  --dropout 0.2 \
  --conv_activation none \
  --fc_activation relu \
  \
  --batch_size 128 \
  --learning_rate 0.002 \
  --inverse 0 \
  --gpu $gpu
