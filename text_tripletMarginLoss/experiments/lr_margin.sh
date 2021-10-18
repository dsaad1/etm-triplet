alpha=0
wandb_name1="moredata_not_softmax_"

python3.8 ../Code/run_model.py \
    --margin 0.1 \
    --lr 0.1 \
    --device 0 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 0.5 \
    --lr 0.1 \
    --device 0 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 1 \
    --lr 0.1 \
    --device 0 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 2 \
    --lr 0.1 \
    --device 1 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 3 \
    --lr 0.1 \
    --device 1 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 5 \
    --lr 0.1 \
    --device 1 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 8 \
    --lr 0.1 \
    --device 2 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

#################################

python3.8 ../Code/run_model.py \
    --margin 0.1 \
    --lr 1 \
    --device 2 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 0.5 \
    --lr 1 \
    --device 2 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 1 \
    --lr 1 \
    --device 4 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 2 \
    --lr 1 \
    --device 4 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 3 \
    --lr 1 \
    --device 4 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 5 \
    --lr 1 \
    --device 5 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 8 \
    --lr 1 \
    --device 5 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

##################################

python3.8 ../Code/run_model.py \
    --margin 0.1 \
    --lr 3 \
    --device 5 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 0.5 \
    --lr 3 \
    --device 6 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 1 \
    --lr 3 \
    --device 6 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 2 \
    --lr 3 \
    --device 6 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 3 \
    --lr 3 \
    --device 7 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 5 \
    --lr 3 \
    --device 7 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 8 \
    --lr 3 \
    --device 7 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &
