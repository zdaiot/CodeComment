EXPNAME=omniglot_resnet_N5_S1
python3 main.py --exp_name $EXPNAME --dataset omniglot --test_N_way 5 --train_N_way 5 --train_N_shots 1 --test_N_shots 1 --batch_size 200  --dec_lr=10000  --iterations 100000 --resnet_pretrained True

EXPNAME=minimagenet_resnet_N5_S1
python3 main.py --exp_name $EXPNAME --dataset mini_imagenet --test_N_way 5 --train_N_way 5 --train_N_shots 1 --test_N_shots 1 --batch_size 200 --dec_lr=15000 --iterations 80000

EXPNAME=omniglot_resnet_N5_S5
python3 main.py --exp_name $EXPNAME --dataset omniglot --test_N_way 5 --train_N_way 5 --train_N_shots 5 --test_N_shots 5 --batch_size 40  --dec_lr=10000  --iterations 80000

EXPNAME=minimagenet_resnet_N5_S5
python3 main.py --exp_name $EXPNAME --dataset mini_imagenet --test_N_way 5 --train_N_way 5 --train_N_shots 5 --test_N_shots 5 --batch_size 40 --dec_lr=15000 --iterations 90000
