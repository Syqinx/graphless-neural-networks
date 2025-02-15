for e in "ogb" "pillow" "scipy" "networkx" "numpy"\
         "tabulate" "tqdm" "PyYAML" "scikit_learn"\
         "googledrivedownloader" "category_encoders"\
         "torch" "dgl" "torchvision" "torchaudio"
do
    # pip show $e
    conda list $e
    echo "---------------------------------"
done