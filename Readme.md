# Nuclear Receptors InteractionGraphNet (NRIGN)
NRIGN: A Unified Deep Graph Model for Discriminating Molecular Types of Ligands Targeting Nuclear Receptors.

# Environment
```
# requirements
python = 3.9.18  
pytorch = 1.13.0  
dgl = 1.1.2+cu117
dgllife = 0.3.2
rdkit = 2023.9.1
scikit-learn = 1.3.2  
numpy = 1.26.4  
pandas = 1.2.4
scipy = 1.11.3  
prefetch_generator = 1.0.3
```

# NRIGN Training (A toy example)
We provide one strategy as example, namely alignment_six, for training.

```bash
nohup python -u ./codes/nrign_train.py --gpuid 0 --epochs 300 --batch_size 128 --lr 0.0003 --l2 0.000001 --dropout 0.15 --dis_threshold 6 > ./result/toy_example.log 2>&1 &
```
We included approximately 200 toy samples in the data folder, due to the large data size, to illustrate the process of training the NRIGN model. Each sample is stored in a pickle file and comprises four RDKit objects: two representing the active and inactive conformations of the protein, and two corresponding to the active and inactive conformations of its ligand.


# Prediction
We use the well-trained NRIGN model to predict the properties of ligands in testset.

```bash
python ./codes/prediction.py --cpu True --num_process 12 --input_path ./data_graph/test/complex 
```

