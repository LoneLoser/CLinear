idx=$1
random_seed=$2
gpu=$3
seq_len=(288 384 528 912)
label_len=(96 192 336 720)
pred_len=(96 192 336 720)

python run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --model CLinear \
  \
  --data ETTh1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --freq h \
  --seq_len ${seq_len[$idx]} \
  --label_len ${label_len[$idx]} \
  --pred_len ${pred_len[$idx]} \
  --period 24 \
  \
  --covar_layers 0 \
  --enc_in 7 \
  --dropout 0.3 \
  --conv_activation none \
  --fc_activation tanh \
  \
  --batch_size 128 \
  --learning_rate 0.001 \
  --inverse 0 \
  --gpu $gpu
