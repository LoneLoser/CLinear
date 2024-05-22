idx=$1
random_seed=$2
gpu=$3
seq_len=(768 768 864 864)
label_len=(96 192 336 720)
pred_len=(96 192 336 720)
lr=(0.001 0.001 0.0001 0.0001)
fc_activation=(tanh tanh tanh relu)

python run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --model CLinear \
  \
  --data ETTm2 \
  --root_path ./data/ETT/ \
  --data_path ETTm2.csv \
  --freq h \
  --seq_len ${seq_len[$idx]} \
  --label_len ${label_len[$idx]} \
  --pred_len ${pred_len[$idx]} \
  --period 96 \
  \
  --covar_layers 0 \
  --enc_in 7 \
  --dropout 0.3 \
  --conv_activation none \
  --fc_activation ${fc_activation[$idx]} \
  \
  --batch_size 128 \
  --learning_rate ${lr[$idx]} \
  --inverse 0 \
  --gpu $gpu
