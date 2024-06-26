# SeNSe

SeNSe: embedding alignment via semantic anchors selection
For details and citations see [References'section](#References)

SeNSe, is an unsupervised method for aligning monolingual embeddings, generating a bilingual dictionary composed of words with the most similar meaning among word vector spaces. This approach selects a seed lexicon of words used in the same context in both corpora without assuming a priori semantic similarities. 
The main strength of SeNSe is the flexibility of the application tied to the robust logic used; indeed, no other works select anchors considering the semantic neighbour of the candidate words.

we describe the building blocks of \sensev, depicted in Fig. \ref{fig:idea}, using the case of BLI. The method readily extends to other alignment scenarios, with the sole distinction that for cross-lingual alignment, a Translator, such as online Google Translate, is required to generate anchor candidate pairs. This step is unnecessary when the embeddings share the same language. The process can be streamlined into five key steps:

![sense_v](https://github.com/filippopallucchini/SeNSe/assets/87646607/5566464c-6a0f-4800-9924-4a09fbaf9907)

- **Vocab of common words**: In this section, we translate all terms from the source embedding into the target language using GT and then compare them via string matching with the dictionary of the target embedding. The result will provide a list of common word pairs between the two vocabularies.
  
- **STEP 1 - Compute SNDCG**: For each couple of the list previously created, we compute a score of semantic similarity using the NDCG score (Normal Discounted Cumulative Gain) named _SNDCG score_ = _Semantic Normal Discounted Cumulative Gain_. The computation of this score involves considering both the cosine similarity between the potential anchor in the target language and the _n_ most similar words of the potential anchor in the source language when translated into the target language (and vice versa: cosine similarity between the potential anchor in the source language and the most similar words of the potential anchor in the target language when translated into the source language), as well as the ranking position of these words. The figure below helps to understand the process just explained.

![example semantic ndcg new](https://github.com/filippopallucchini/SeNSe/assets/87646607/04126e09-371a-4c07-87d3-ec3cd20ad434)

- **STEP 2 - Select Best Anchors**: We carefully choose the Best Anchors by retaining only those with a specific threshold value of _SNDCG score_. Additionally, to account for cases where two source anchors might have the same translation, we deduplicate the list by selecting the pair of anchors with the highest _SNDCG score_.
  
- **STEP 3 - Dispersion**: To ensure a well-balanced list of anchors in the embedding space, we choose several that are evenly distributed. Our goal is to prevent the alignment from being biased towards specific points in the vector space merely due to the prevalence of highly similar terms. To accomplish this, we meticulously pinpoint closely located anchors within the space and selectively retain the most qualitative ones, thereby maximizing their _SNDCG scores_. By the conclusion of **STEP 3**, we acquire the **seed lexicon**, constituting the definitive list of anchors for executing the alignment.

- **STEP 4 - Orthogonal Mapping Methods**: The output of the previous step is the Seed Lexicon that will be used for the alignment through Orthogonal Procrustes (OP). The goal of OP is to learn an orthogonal transformation matrix _Q_, that closest maps source embedding (A) to target embedding (B). It has been shown that this problem accepts a closed-form solution via Singular Value Decomposition (SVD). The orthogonality of matrix $_Q_ ensures that _AQ_ undergoes only unitary transformations, such as reflection and rotation, thereby preserving the inner product between its word vectors.

- **STEP 5 - Alignment Evaluation**: This last step is necessary to evaluate the goodness of the alignment.

## **Deployment**

Use Python 3.8.10

1. Upload embeddings that you want to align in the 'input' folder. E.g. we use in our experiments embeddings from [Dinu et al.](https://wiki.cimec.unitn.it/tiki-index.php?page=CLIC). You can use the get_data.sh from [Artetxe et al.](https://github.com/artetxem/vecmap/tree/master).
2. Upload the file with stopwords in the 'utils' folder, if needed.
3. Run CREATE_TRANSLATION_DICTIONARY.py
4. Run CREATE_DICT_MOST_SIMILAR.py
5. Run ALIGNMENT.py

## References
```
@article{malandri2024sense,
  title={SeNSe: embedding alignment via semantic anchors selection},
  author={Malandri, Lorenzo and Mercorio, Fabio and Mezzanzanica, Mario and Pallucchini, Filippo},
  journal={International Journal of Data Science and Analytics},
  year={2024},
  publisher={Springer}
}
```
