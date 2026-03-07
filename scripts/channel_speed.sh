
echo "===> start preProcess data"
python preProcessing/convert_processed_to_astgcn.py --processed_dir ./data/processed --base_graph_path ./data/PEMS04/PEMS04.npz --num_of_hours 1 --num_of_days 0 --num_of_weeks 0 --input_feature_index 2

echo "===> start train model"

python train_ASTGCN_r.py --config configurations/PEMS04_astgcn.conf