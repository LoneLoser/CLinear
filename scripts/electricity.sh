idx=$1
random_seed=$2
gpu=$3
seq_len=(288 384 528 912)
label_len=(96 192 336 720)
pred_len=(96 192 336 720)
lr=(0.002 0.001 0.001 0.001)

python run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --model CLinear \
  \
  --data custom \
  --root_path ./data/electricity/ \
  --data_path electricity.csv \
  --freq h \
  --seq_len ${seq_len[$idx]} \
  --label_len ${label_len[$idx]} \
  --pred_len ${pred_len[$idx]} \
  --period 24 \
  \
  --covar_layers 0 \
  --enc_in 321 \
  --dropout 0.2 \
  --conv_activation none \
  --fc_activation relu \
  \
  --batch_size 16 \
  --learning_rate ${lr[$idx]} \
  --inverse 0 \
  --gpu $gpu
