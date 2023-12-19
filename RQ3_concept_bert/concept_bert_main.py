# This pipeline receives as input a sentence containing a masked word
# It outputs ranked completions (unlike other token-level algorithms, our approach is concept-based)
# Steps:
# Input: a sentence containing a masked word
# 0. Preprocess the input sentence
# 1. Retrieve the top-k completions according to BERT
# 2. Augmentation (by paraphrasing)
#   - Replace the masked word with BERT's first completion
#   - Use word-tune to create paraphrasing of the input sentence
#   - Mask all sentences
# 3. Use BERT to retrieve the top k possible completions and save their weights and conditional embeddings
# 4. Clean
#   4.0 Drop augmentations according to rank vectors (if drop_aug=True)
#   4.1 Drop sparse completions
# 5. Reduce the dimensionality
# 6. Cluster the remaining completions
# 7. Rank the clusters using the weights of the completions they contain

# # # IMPORTS # # #
import argparse

from transformers import BertTokenizer, BertForMaskedLM, BertModel
from nltk.corpus import stopwords
from textblob import TextBlob
import requests
from tqdm import tqdm
import pickle

from RQ3_concept_bert.concept_bert_utilities import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="bert-base-uncased",
        type=str,
        required=False,
        help="Which BERT model to use."
    )
    parser.add_argument(
        "--bert_top_k",
        default=100,
        type=int,
        required=False,
        help="How many token completions from the top of BERT's ranked "
             "vocabulary to use for the concept-aware manipulation.",
    )
    parser.add_argument(
        "--set_name",
        default="train",
        # choices=["train", "test", "dev", "our_manual", "amazon_apparel_10k_head"],
        type=str,
        required=False,
        help="Which dataset to use. "
             "Options are (unless use a new dataset): [train, test, dev, our_manual, amazon_apparel_10k_head]"
    )
    parser.add_argument(
        "--cluster_algo",
        default="AgglomerativeClustering",
        choices=["AgglomerativeClustering", "DBSCAN"],
        type=str,
        required=False,
        help="Which clustering algorithm to use. Options are: [AgglomerativeClustering, DBSCAN]"
    )
    parser.add_argument(
        "--drop_ratio",
        default=0.45,
        type=float,
        required=False,
        help="Determines the amount of token completions to drop (due to sparsity). A function of #paraphrases.",
    )
    parser.add_argument(
        "--dist_func",
        default="cosine_similarity",
        choices=["cosine_similarity", "cosine_similarity", "pairwise_distances", "euc_rank", "min_aug"],
        type=str,
        required=False,
        help="Which distance function to use for the clustering. "
             "Possible options are: [cosine_similarity, cosine_similarity, pairwise_distances, euc_rank, min_aug]",
    )
    parser.add_argument(
        "--dim_red_algo",
        default="comb",
        choices=["comb", "umap", "pca", "tsne"],
        type=str,
        required=False,
        help="Which dimensionality reduction algorithm to use (combination of PCA and tSNE, UMAP, PCA, or tSNE)."
             "Possible options are: [comb, umap, pca, tsne]",
    )
    parser.add_argument(
        "--dim_red_n_comp",
        default=30,
        type=int,
        required=False,
        help="The dimension to reduce the data to.",
    )
    parser.add_argument(
        "--rank_weight",
        default=0,
        type=int,
        required=False,
        help="How much to consider the weights of the cluster-tokens when ranking the clusters.",
    )
    parser.add_argument(
        "--dbscan_eps_mult",
        default=0.1,
        type=float,
        required=False,
        help="The required epsilon (a function of the distance matrix of the data).",
    )
    parser.add_argument(
        "--dbscan_min_samples",
        default=3,
        type=int,
        required=False,
        help="The minimal number of data samples to form a cluster for DBSCAN.",
    )
    parser.add_argument(
        "--alpha",
        default=0.7,
        type=float,
        required=False,
        help="The weights on BERT's weights for the token clusters (relative to # of repetitions).",
    )
    parser.add_argument(
        "--drop_aug",
        default=False,
        action="store_true",
        required=False,
        help="Whether to drop paraphrases that have very different completions from the others."
    )
    parser.add_argument(
        "--scatter_plot",
        default=False,
        action="store_true",
        required=False,
        help="Whether to create a 2D plot (dim. red. using t-SNE) of the embedding space for visualisation."
    )

    return parser.parse_args()


# # # # # # MAIN # # # # # #
def main(args):

    # # # PRE-LOADS # # #
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    pred_mlm = BertForMaskedLM.from_pretrained(args.model_name)
    pred_mlm.eval()

    embd_lm = BertModel.from_pretrained(args.model_name, output_hidden_states=True)
    embd_lm.eval()

    stopwords_set = set(stopwords.words('english'))

    # # # # # # # # # # #
    # # # # Step 0  # # # #
    print("step 0")  # Load and preprocess the data.
    # # # # Step 0  # # # #

    df = get_df(args.set_name)

    with open(f'output4evaluation\\bert\\{args.set_name}\\config.txt', 'w') as f:
        f.write(f'bert_top_k={args.bert_top_k},\ndrop_ratio={args.drop_ratio}_'
                f'cluster_algo={args.cluster_algo}_'
                f'rank_weight={args.rank_weight}_drop_aug={args.drop_aug}'
                f'_dist_func={args.dist_func}_dim_red_n={args.dim_red_n_comp}'
                f'_dim_algo={args.dim_red_algo}_alpha={args.alpha},'
                f'model={args.model_name}')

    for sen_num, sentence in enumerate(tqdm(df.paraphrased)):
        if type(sentence) != str or len(sentence) < 5 or len(sentence.split(" ")) < 4:
            continue

        blob = TextBlob(sentence)
        if len(blob.raw_sentences) != 1:
            continue

        if args.set_name == "amazon_apparel_10k_head":
            nouns = [tup[0] for tup in blob.tags if tup[1] in {"NN", "NNS", "NNP", "NNSP"}
                     and tup[0].isalpha() and sentence.count(tup[0]) == 1]
            print(sentence)
            try:
                sentence = sentence.replace(nouns[0], "[MASK]")
            except IndexError:
                continue
        else:
            sentence = sentence.replace("<extra_id_0>", "[MASK]")

        if "[MASK]" not in sentence:
            continue
        print(sen_num, sentence)

        # # # # # # # # # # #
        # # # # Step 1  # # # #
        print("step 1")  # Retrieve the top-k completions according to BERT.
        # # # # Step 1  # # # #

        top_comp = get_top_comp(sentence, pred_mlm, embd_lm, tokenizer, stopwords_set)

        # # # # # # # # # # #
        # # # # Step 2  # # # #
        print("step 2")  # Replace the [MASK] token with the first completion and use it to find paraphrases.
        # # # # Step 2  # # # #

        completed_sentence = sentence.replace("[MASK]", top_comp)
        print(f"The completed sentence is: {completed_sentence}")

        # wordtune API (https://www.wordtune.com/#rewrite-demo)
        resp = requests.post(
            "https://api.ai21.com/studio/v1/experimental/rewrite",
            headers={"Authorization": "Bearer OVLceQi85TTTlXePkctHI0V4uWoAkQXw"},  # TODO
            json={"text": completed_sentence, "intent": "general"})
        try:
            paraphrases = [compl_rep_with_extra_id(l["text"], top_comp)
                           for l in resp.json()["suggestions"]
                           if top_comp in l["text"]]
        except KeyError:
            continue
        if len(paraphrases) == 0:
            continue
        print(f"{len(paraphrases)} paraphrases were found.")
        paraphrases.insert(0, sentence)

        # # # # # # # # # # #
        # # # # Step 3  # # # #
        print("step 3")  # Retrieve and aggregate for all paraphrases the ranked list
        # of completions along with their corresponding weights, embeddings, and rankings.
        # # # # Step 3  # # # #

        completions_agg, weights_agg, embeddings_agg, rankings_agg, completions_counter = get_aggr_data(
            paraphrases, pred_mlm, embd_lm, tokenizer, topk=args.bert_top_k)

        # # # # # # # # # # #
        # # # # Step 4  # # # #
        print("step 4")  # Drop odd paraphrases (if true) and sparse completions.
        # # # # Step 4  # # # #
        # Step 4.0 #

        # Dropping odd augmentations
        if args.drop_aug:
            completions_agg, weights_agg, embeddings_agg, rankings_agg = drop_aug(
                completions_agg, weights_agg, embeddings_agg, rankings_agg, len(paraphrases), args.bert_top_k)

        # Step 4.1 #
        # Dropping sparse completions
        completions_list, embeddings_list, completion2weight, completion2embeddings, completion2rank, completion2rep = \
            drop_sparse_completions(int(len(paraphrases)*args.drop_ratio), completions_counter, stopwords_set,
                                    completions_agg, weights_agg, embeddings_agg, rankings_agg)

        # # # # # # # # # # #
        # # # # Step 5 # # # #
        print("step 5")  # Reduce the embedding dimensionality.
        # # # # Step 5 # # # #

        if args.scatter_plot:  # For visualization in 2D
            scatter_plot(embeddings_list, completions_list, sentence)

        embeddings_agg_reduced, completion2embeddings_red = get_embd_dim_red(
            args.dim_red_algo, embeddings_list, completions_agg, args.dim_red_n_comp)

        # # # # # # # # # # #
        # # # # Step 6  # # # #
        print("step 6")  # Cluster the remaining tokens.
        # # # # Step 6  # # # #

        if args.cluster_algo == "DBSCAN":
            clusters, dbscan_eps_mult_new = get_dbscan_clusters(
                                                                completion2embeddings_red,
                                                                completions_list,
                                                                completion2rank,
                                                                args.dbscan_eps_mult,
                                                                args.dbscan_min_samples,
                                                                args.rank_weight,
                                                                dist_func=args.dist_func)
        elif args.cluster_algo == "AgglomerativeClustering":
            clusters, normalized_dist_mat, trained_cluster_algo = get_agglo_clusters(
                                                                                     completion2embeddings_red,
                                                                                     completions_list,
                                                                                     completion2rank,
                                                                                     args.rank_weight,
                                                                                     args.dist_func)

            plot_dendrogram(trained_cluster_algo, sentence,
                            f'output4evaluation\\bert\\{args.set_name}\\{sen_num}.png', completions_list)
        cluster2word_dict, cluster2weight_dict, cluster2rank_dict, cluster2rep_dict, cluster2best_comp_dict = \
            cluster_processing(clusters, completions_list, completion2weight,
                               completion2rank, completion2rep, args.bert_top_k, args.alpha)

        # Save Concept-BERT's clusters
        print(f"bert_top_k={args.bert_top_k}, drop_ratio={args.drop_ratio}, cluster_algo={args.cluster_algo}, "
              f"rank_weight={args.rank_weight}, dbscan_eps_mult={dbscan_eps_mult_new}, "
              f"dbscan_min_samples={args.dbscan_min_samples}")

        # Save BERT's completions
        top_k_completions, completions_proba, _ = get_bert_completion(
            sentence, pred_mlm, embd_lm, tokenizer, topk=args.bert_top_k)

        with open(f'output4evaluation\\bert\\{args.set_name}\\{sen_num}_bert_plain_lm_'
                  f'dbscan_eps_mult={args.dbscan_eps_mult_new}_'
                  f'dbscan_min_samp={args.dbscan_min_samples}.pickle', 'wb') as fp:
            pickle.dump(top_k_completions, fp, protocol=pickle.HIGHEST_PROTOCOL)

        # # # # # # # # # # #
        # # # # Step 7  # # # #
        print("step 7")  # Rank the clusters and plot them.
        # # # # Step 7  # # # #

        keys = list(cluster2word_dict.keys())
        weights = [args.alpha*cluster2weight_dict[label_key]
                   + (1-args.alpha)*cluster2rep_dict[label_key] for label_key in keys]
        sorted_keys = [label_key for _, label_key in sorted(zip(weights, keys), reverse=True)]

        sorted_cluster2word_dict = get_ranked_clusters(
            sorted_keys, completion2embeddings, cluster2word_dict, cluster2best_comp_dict, top_k_completions)

        print("Done with the sentence\nSaving the completions")

        try:
            with open(f'output4evaluation\\bert\\{args.set_name}\\{sen_num}_bert_clusters_'
                      f'dbscan_eps_mult={dbscan_eps_mult_new}_'
                      f'dbscan_min_samp={args.dbscan_min_samples}.pickle', 'wb') as fp:
                pickle.dump(sorted_cluster2word_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        except FileNotFoundError:
            print(f"File error for sen num {sen_num}: {sentence}")


if __name__ == "__main__":
    main(parse_args())
