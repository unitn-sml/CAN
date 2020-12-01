cd ../MolGAN
python example.py --batch_size 32
--dropout 0.0
--n_critic 5
--metrics logp,sas,qed
--n_samples 5000
--z_dim 37
--lambd 0.0
--epochs 300
--activation_epoch 150
--activation_epoch_SL 0
--save_every 10
--lr 0.001
--batch_discriminator True
--name ./output/final_SL_metrics1
--sl_use_sigmoid False
--discrete_z 5