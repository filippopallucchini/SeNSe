# SeNSe
SeNSe: embedding alignment via semantic anchors selection

Use Python 3.8.10

1. Upload embeddings that you want to align in the 'input' folder. E.g. we use in our experiments embeddings from Dinu et al. (https://wiki.cimec.unitn.it/tiki-index.php?page=CLIC). You can use the get_data.sh from Artetxe et al. https://github.com/artetxem/vecmap/tree/master .
2. Upload the file with stopwords in 'utils' folder, if needed.
3. Run CREATE_TRANSLATION_DICTIONARY.py
4. Run CREATE_DICT_MOST_SIMILAR.py
5. Run ALIGNMENT.py
