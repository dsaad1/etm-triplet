alpha=0
wandb_name1="moredata_not_softmax_"

##################################

python3.8 ../Code/run_model.py \
    --margin 0.1 \
    --lr 8 \
    --device 0 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 0.5 \
    --lr 8 \
    --device 0 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 1 \
    --lr 8 \
    --device 0 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 2 \
    --lr 8 \
    --device 1 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 3 \
    --lr 8 \
    --device 1 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 5 \
    --lr 8 \
    --device 1 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 8 \
    --lr 8 \
    --device 2 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

#################################

python3.8 ../Code/run_model.py \
    --margin 0.1 \
    --lr 15 \
    --device 4 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 0.5 \
    --lr 15 \
    --device 4 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 1 \
    --lr 15 \
    --device 5 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 2 \
    --lr 15 \
    --device 5 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 3 \
    --lr 15 \
    --device 6 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 5 \
    --lr 15 \
    --device 6 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 8 \
    --lr 15 \
    --device 7 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

#################################

python3.8 ../Code/run_model.py \
    --margin 0.1 \
    --lr 200 \
    --device 7 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 0.5 \
    --lr 200 \
    --device 2 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 1 \
    --lr 200 \
    --device 2 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 2 \
    --lr 200 \
    --device 4 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 3 \
    --lr 200 \
    --device 5 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 5 \
    --lr 200 \
    --device 6 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &

python3.8 ../Code/run_model.py \
    --margin 8 \
    --lr 200 \
    --device 7 \
    --alpha $alpha \
    --wandb_name $wandb_name1 &