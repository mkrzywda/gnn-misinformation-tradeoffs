#!/bin/bash

# Skrypt działa w bieżącej ścieżce (pwd)

# Lista folderów do utworzenia
FOLDERS=("appnp" "gin" "gcn" "gat" "graphsage" "sgc" "jknet" "tagconv" "feastconv" "chebnet")

# Utwórz foldery
for folder in "${FOLDERS[@]}"; do
    mkdir -p "$folder"
done

# Przenoszenie plików do odpowiednich folderów
declare -A MAPPING

# Generowanie wzorców dla każdego datasetu i modelu
DATASETS=("kaggle" "liar" "mpid" "welfake" "fakenewsnet")
MODELS=("appnp" "gin" "gcn" "gat" "graphsage" "sgc" "jknet" "tagconv" "feastconv" "chebnet")

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        pattern="${dataset}_${model}_fold*"
        MAPPING["$pattern"]="$model"
    done
done

for pattern in "${!MAPPING[@]}"; do
    target_folder="${MAPPING[$pattern]}"
    mv $pattern "$target_folder" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "Przeniesiono $pattern do $target_folder"
    fi
done

echo "Przenoszenie zakończone."

