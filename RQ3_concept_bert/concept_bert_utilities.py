import torch
import transformers
from scipy.cluster.hierarchy import dendrogram, leaves_list, ward
from scipy.spatial import distance
import umap
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd


# # # FUNCTIONS # # #

# # # # Step 0  # # # #
def get_df(set_name: str) -> pd.DataFrame:
    """
    Load the csv as dataframe.
    """

    if set_name == "amazon_apparel_10k_head":
        df = pd.read_csv(f"amazon_reviews_data\\amazon_reviews_us_Apparel_v1_00_head_10k.csv")
        df["paraphrased"] = df["review_body"]
    else:
        df = pd.read_csv(f"protoqa_paraphrases_{set_name}set.csv")
        return df


# # # # Step 1 (& 3 & 6)  # # # #
def get_bert_completion(sentence: str, pred_mlm, embd_lm, tokenizer, topk: int) -> str:
    """
    Returns the top-k completions according to the BERT model
    along with their corresponding weights and contextual embeddings.
    """

    text = "[CLS] %s [SEP]" % sentence
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    # Predict all tokens
    with torch.no_grad():
        outputs = pred_mlm(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, topk, sorted=True)
    if topk == 1:
        completion = tokenizer.convert_ids_to_tokens([top_k_indices])[0]
        return sentence.replace("[MASK]", completion)

    else:
        completions, embeddings = [], []
        for i, pred_idx in enumerate(top_k_indices):
            # if top_k_weights[i] >= weight_threshold:
            predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
            completions.append(predicted_token)
            # Extract the conditional completion embedding
            embeddings.append(_get_embd(embd_lm, tokenizer, masked_index,
                                        tokenized_text, predicted_token).numpy())

        return completions, top_k_weights, embeddings


# # # # Step 1 (& 3 & 6)  # # # #
def _get_embd(model, tokenizer, masked_index: int,
              tokenized_text: list, predicted_token: str) -> list:
    """
    Returns the contextual embeddings of the masked index.
    """

    tokenized_text[masked_index] = tokenizer.tokenize(predicted_token)[0]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)

    # Return embedding only for the predicted token
    return token_embeddings[masked_index]


# # # # Step 1  # # # #
def get_top_comp(sentence: str, pred_mlm: transformers.BertForMaskedLM, embd_lm: transformers.BertModel,
                 tokenizer: transformers.BertTokenizer, stopwords_set: set) -> str:
    """
    returns the first completion that is not filtered according to _filter_comp.
    """

    top_k_completions, _, _ = get_bert_completion(sentence, pred_mlm, embd_lm, tokenizer, topk=200)
    try:
        top_comp = _filter_comp(top_k_completions, stopwords_set)[0]
    except IndexError:
        top_k_completions, _, _ = get_bert_completion(sentence, pred_mlm, embd_lm, tokenizer, topk=1000)
        top_comp = _filter_comp(top_k_completions, stopwords_set)[0]
    return top_comp


# # # # Step 1  # # # #
def _filter_comp(completions: list, stopwords_set: set) -> list:
    """
    Filters tokens that have less than 3 characters or that are in the stopwords_set.
    """

    return [tok for tok in completions if len(tok) > 3 and tok not in stopwords_set]


# # # # Step 2  # # # #
def compl_rep_with_extra_id(sentence: str, comp: str) -> str:
    """
    Replaces comp in the sentence with "[MASK]".
    """

    try:
        space_end_ind = sentence[sentence.index(comp) + len(comp):].index(" ") + sentence.index(comp) + len(comp)
        return f"{sentence[:sentence.index(comp)]}[MASK]{sentence[space_end_ind:]}"

    except ValueError:
        space_end_ind = len(sentence)
        return f"{sentence[:sentence.index(comp)]}[MASK]{sentence[space_end_ind:]}."


# # # # Step 3  # # # #
def get_aggr_data(paraphrases: list, pred_mlm: transformers.BertForMaskedLM, embd_lm: transformers.BertModel,
                  tokenizer: transformers.BertTokenizer, topk) -> ([], [], [], [], defaultdict):
    """
    Aggregates the completions and their corresponding weights, embeddings and rankings for all the paraphrases.
    """

    completions_counter = defaultdict(int)  # Counting how many times each completion appeared (for step #4)
    completions_agg, weights_agg, embeddings_agg, rankings_agg = [], [], [], []
    for sen in paraphrases:
        completions, weights, embeddings = get_bert_completion(
            sen, pred_mlm, embd_lm, tokenizer, topk)
        completions_agg += completions
        weights_agg += weights
        embeddings_agg += embeddings
        rankings_agg += list(np.arange(len(completions)))
        completions_counter = _counter_update(completions_counter, completions)
    return completions_agg, weights_agg, embeddings_agg, rankings_agg, completions_counter


# # # # Step 3  # # # #
def _counter_update(completions_counter: defaultdict, completions: list) -> defaultdict:
    """
    Counts the repetitions of each completion across the different augmentations.
    """

    for c in completions:
        completions_counter[c] += 1

    return completions_counter


# # # # Step 4  # # # #
def drop_aug(completions_agg: list, weights_agg: list, embeddings_agg: list, rankings_agg: list,
             len_paraphrases: int, top_k: int) -> ([], [], [], []):
    """
    Drops paraphrases (and their completions) that are too odd compared to the rest.
    """

    ranks_bow_matrix = _get_rank_mat(completions_agg, len_paraphrases, top_k)
    augmentations_ind2drop = _get_drop_aug_ind(ranks_bow_matrix)

    for ind in augmentations_ind2drop:
        completions_agg, weights_agg, embeddings_agg, rankings_agg = _drop_by_ind(
            completions_agg, weights_agg,
            embeddings_agg, rankings_agg,
            list(np.arange(ind * top_k,
                           (ind + 1) * top_k)))
    print(f"{len(augmentations_ind2drop)} paraphrases removed.")
    return completions_agg, weights_agg, embeddings_agg, rankings_agg


# # # # Step 4  # # # #
def _get_rank_mat(completions_agg: list, num_paraphrases: int, bert_top_k: int) -> np.ndarray:
    """
    Used to drop odd augmentations.
    Creates a matrix of the completion-ranks.
    Each row is the rank of a completion according to the different paraphrases. Each column represents the paraphrases.
    """

    augmentations_bow_dict = {c: i for i, c in
                              enumerate(completions_agg[:bert_top_k + 1])}  # align according to the input sen
    ranks_bow_matrix = np.zeros((num_paraphrases, bert_top_k))
    for i in range(num_paraphrases):
        rank = []
        for c in completions_agg[i * bert_top_k:(i + 1) * bert_top_k]:
            try:
                rank.append(augmentations_bow_dict[c])
            except KeyError:
                rank.append(bert_top_k)
        ranks_bow_matrix[i, :] = rank

    return ranks_bow_matrix


# # # # Step 4  # # # #
def _get_drop_aug_ind(ranks_bow_matrix: np.ndarray) -> list:
    """
    Used to drop odd augmentations.
    Returns the augmentation indices to drop.
    """

    sim_matrix = cosine_similarity(ranks_bow_matrix)
    augmentations_ind2drop = []
    for i, sim_score in enumerate(sim_matrix[:, 0]):
        if sim_score < np.mean(sim_matrix[:, 0]):
            augmentations_ind2drop.append(i)
    augmentations_ind2drop.sort(reverse=True)

    return augmentations_ind2drop


# # # # Step 4  # # # #
def drop_sparse_completions(n_drop: int, completions_counter: defaultdict, stopwords_set: set, completions_agg: list,
                            weights_agg: list, embeddings_agg: list, rankings_agg: list) \
                            -> ([], [], [], [], defaultdict):
    """
    Drops sparse token completions.
    """

    # Finding which completions to drop
    com2drop = _get_drop_comp(n_drop, completions_counter, stopwords_set)
    # Actually dropping the completions
    ind_drop = _get_drop_ind(com2drop, completions_agg)  # Finding the indices to drop
    if len(ind_drop) > 0:  # Dropping according the indices
        completions_agg, weights_agg, embeddings_agg, rankings_agg = _drop_by_ind(completions_agg, weights_agg,
                                                                                  embeddings_agg, rankings_agg,
                                                                                  ind_drop)

    # Merging rank, embedding and weight for repeated completions
    completions_agg, weights_agg, embeddings_agg, rankings_agg = _sorting(completions_agg, weights_agg,
                                                                          embeddings_agg, rankings_agg)

    return _merge_dup_completions(completions_agg, weights_agg, embeddings_agg, rankings_agg)


# # # # Step 4  # # # #
def _get_drop_comp(n: int, completions_counter: defaultdict, stopwords_set: set) -> set:
    """
    Used to drop sparse completions.
    Returns the completions to drop according to n and completions_counter.
    """

    com2drop = set()
    for c in completions_counter.keys():
        if completions_counter[c] < n or len(c) < 3 or c in stopwords_set:
            com2drop.add(c)

    return com2drop


# # # # Step 4  # # # #
def _get_drop_ind(com2drop: set, completions: list) -> list:
    """
    Used to drop sparse completions.
    Returns the token indices to drop.
    """

    ind_drop = []
    for i, c in enumerate(completions):
        if c in com2drop or "##" in c:
            ind_drop.append(i)
    ind_drop.sort(reverse=True)

    return ind_drop


# # # # Step 4  # # # #
def _drop_by_ind(completions_agg: list, weights_agg: list, embeddings_agg: list, rankings_agg: list, ind_drop: list)\
        -> [list, list, list, list]:
    """
    Used to drop sparse completions.
    Returns the dropped arrays.
    """

    ind_drop.sort(reverse=True)
    for ind in ind_drop:
        del(completions_agg[ind])
        del(weights_agg[ind])
        del(embeddings_agg[ind])
        del(rankings_agg[ind])

    return completions_agg, weights_agg, embeddings_agg, rankings_agg


# # # # Step 4  # # # #
def _sorting(completions_agg: list, weights_agg: list, embeddings_agg: list, rankings_agg: list)\
        -> [list, list, list, list]:
    """
    Used to drop sparse completions.
    Sorts and aggregates the arrays after dropping elements from them.
    """

    ind = sorted(range(len(completions_agg)), key=lambda k: completions_agg[k])
    completions_agg_sorted = [completions_agg[i] for i in ind]
    weights_agg_sorted = [weights_agg[i] for i in ind]
    embeddings_agg_sorted = [embeddings_agg[i] for i in ind]
    rankings_agg_sorted = [rankings_agg[i] for i in ind]

    return completions_agg_sorted, weights_agg_sorted, embeddings_agg_sorted, rankings_agg_sorted


# # # # Step 4  # # # #
def _merge_dup_completions(completions_agg: list, weights_agg: list, embeddings_agg: list, rankings_agg: list)\
        -> [list, list, defaultdict, defaultdict, defaultdict, defaultdict]:
    """
    Used to drop sparse completions.
    Merges duplicate token completions (and their corresponding embeddings, weights, ranks and repetitions).
    Aggregation is done as following:
    - Embeddings: The contextual embedding using the original input sentence.
    - Weights: The mean weight across the different paraphrases.
    - Ranks: The minimal rank (lower means more probable) across the different paraphrases.
    - Repetitions: The # of times a completion appeared across the different paraphrases (max. value = |paraphrases|).
    """

    prev = completions_agg[0]
    c = 0
    completions_list, embeddings_list, completion2embeddings, completion2weight, completion2rank, completion2rep = \
        [], [], defaultdict(list), defaultdict(), defaultdict(), defaultdict()
    embeddings_agg = np.array(embeddings_agg)
    for i, element in enumerate(completions_agg):
        if element == prev:
            completion2embeddings[element].append(np.array(embeddings_agg[i]))
            embeddings_list.append(np.array(embeddings_agg[i]))
            if i == len(completions_agg) - 1:
                completions_list.append(prev)
                completion2weight[prev] = np.mean(weights_agg[c:i+1])
                completion2rank[prev] = np.min(rankings_agg[c:i+1])
                completion2rep[prev] = i-c+1
        else:
            completions_list.append(prev)
            embeddings_list.append(np.array(embeddings_agg[i]))
            completion2weight[prev] = np.mean(weights_agg[c:i])
            completion2rank[prev] = np.min(rankings_agg[c:i])
            completion2rep[prev] = i-c
            prev = element
            completion2embeddings[element].append(np.array(embeddings_agg[i]))
            c = i

    return completions_list, embeddings_list, completion2weight, completion2embeddings, completion2rank, completion2rep


# # # # Step 5 # # # #
def scatter_plot(embeddings_list: list, completions_list: list, sentence: str):
    """
    Creates a 2D scatter plot of the embedding space using t-SNE for dimensionality reduction (n=2). Used for intuition.
    """

    embeddings_agg_reduced = TSNE(n_components=2, init='pca', perplexity=10). \
        fit_transform(np.array(embeddings_list))

    fig, ax = plt.subplots()
    ax.scatter(embeddings_agg_reduced[:, 0], embeddings_agg_reduced[:, 1])

    for i, txt in enumerate(completions_list):
        ax.annotate(txt, (embeddings_agg_reduced[i, 0], embeddings_agg_reduced[i, 1]))

    plt.title(f"t-SNE with N=2 for:\n{sentence}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


# # # # Step 5 # # # #
def get_embd_dim_red(dim_red_algo: str, embeddings_list: list, completions_agg: list, dim_red_n_comp: int):
    """
    Used to drop sparse completions.
    Sorts and aggregates the arrays after dropping elements from them.
    """

    if dim_red_algo == "comb":
        embeddings_agg_reduced = comb_dim_red(embeddings_list, n_components=dim_red_n_comp)
    else:
        embeddings_agg_reduced = dim_red(
            embeddings_list, dim_red_algo=dim_red_algo, n_components=dim_red_n_comp)

    completion2embeddings_red = defaultdict(list)
    for i, key in enumerate(completions_agg):
        completion2embeddings_red[key].append(embeddings_agg_reduced[i])

    return embeddings_agg_reduced, completion2embeddings_red


# # # # Step 5 # # # #
def comb_dim_red(embeddings_agg: list, n_components: int) -> sklearn.manifold.TSNE:
    """
    Reduces the data dimensionality using PCA (n=min(100, # embeddings))
    and then using t-SNE if n_components<pca_red.shape[0].
    """

    pca_red = PCA(n_components=np.min([100, len(embeddings_agg)]), svd_solver='full').\
        fit_transform(np.array(embeddings_agg))

    return TSNE(n_components=np.min([n_components, pca_red.shape[0]]), init='pca',
                perplexity=np.min([10, pca_red.shape[0]]), method='exact').fit_transform(pca_red)


# # # # Step 5 # # # #
def dim_red(embeddings_agg: list, dim_red_algo: str, n_components: int) -> sklearn.manifold:
    """
    Reduces the data dimensionality using the chosen algorithm and n_components.
    """

    if dim_red_algo == "umap":
        return umap.UMAP(n_components=n_components).fit_transform(np.array(embeddings_agg))  # n_components = 3

    elif dim_red_algo == "pca":
        return PCA(n_components=n_components, svd_solver='full').fit_transform(
            np.array(embeddings_agg))  # n_components = 0.95

    elif dim_red_algo == "tsne":
        return TSNE(n_components=n_components, init='pca', perplexity=10).fit_transform(np.array(embeddings_agg))

    else:
        print(f"Dim. red. algorithm {dim_red_algo} was not found.\nOptions: [umap, pca, tsne].")
        exit()


# # # # Step 5 # # # #
def get_agglo_clusters(embeddings_agg: dict, completions_agg: list,
                 rankings_agg: dict, rank_weight=0.0, dist_func="pairwise_distances")\
        -> [list, np.ndarray, sklearn.cluster]:
    """
    Returns the Agglomerative clusters with the normalized distance matrix and the fitted clustering algorithm.
    """

    dist_mat = _dist_func(embeddings_agg, completions_agg, rankings_agg, rank_weight=rank_weight, dist_func=dist_func)
    normalized_dist_mat = (dist_mat - np.min(dist_mat)) / (np.max(dist_mat) - np.min(dist_mat))
    cluster_algo = AgglomerativeClustering(affinity="precomputed", linkage="complete", distance_threshold=0.45,
                                               n_clusters=None, compute_distances=True)
    clusters = cluster_algo.fit_predict(normalized_dist_mat)

    return clusters, normalized_dist_mat, cluster_algo


# # # # Step 6  # # # #
def plot_dendrogram(model, sentence: str, path: str, completions_list: list):
    """
    Plots and saves the dendrogram of the Agglomerative clustering algorithm.
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig = plt.figure(figsize=(3000 * px, 1600 * px))
    dendo_dict = dendrogram(linkage_matrix, get_leaves=True)
    labels = [completions_list[i] for i in dendo_dict["leaves"]]
    xmin, xmax = np.array(dendo_dict["icoord"]).min(), np.array(dendo_dict["icoord"]).max()
    plt.xticks(ticks=np.arange(start=xmin, step=10, stop=xmax+1), labels=labels, fontsize=15)
    plt.title(sentence)
    plt.savefig(f"{path}.png")
    plt.show()


# # # # Step 6  # # # #
def _dist_func(embeddings_agg: dict, completions_agg: list, rankings_agg: dict,
               rank_weight=0.0, dist_func="pairwise_distances") -> np.ndarray:
    """
    Calculates the normalized distance matrix.
    """

    if dist_func == "cosine_similarity":
        mean_embeddings_agg = []
        for c in completions_agg:
            mean_embeddings_agg.append(np.mean(np.array(embeddings_agg[c]), axis=0))
        return -cosine_similarity(np.array(mean_embeddings_agg), np.array(mean_embeddings_agg))

    if dist_func == "pairwise_distances":
        mean_embeddings_agg = []
        for c in completions_agg:
            mean_embeddings_agg.append(np.mean(np.array(embeddings_agg[c]), axis=0))
        return pairwise_distances(np.array(mean_embeddings_agg), np.array(mean_embeddings_agg))

    elif dist_func == "min_aug":
        return _minimal_augm_dist_fun(embeddings_agg, completions_agg)

    elif dist_func == "euc_rank":
        mean_embeddings_agg, min_rank_embeddings_agg = [], []
        for c in completions_agg:
            mean_embeddings_agg.append(np.mean(np.array(embeddings_agg[c]), axis=0))
            min_rank_embeddings_agg.append(np.min(np.array(rankings_agg[c]), axis=0))
        return _euc_dist_and_rank_fun(mean_embeddings_agg, min_rank_embeddings_agg, lambda_param=rank_weight)

    else:
        print(f"Distance function {dist_func} was not found.\nOptions: [pairwise_distances, min_aug, euc_rank].")
        exit()


# # # # Step 6  # # # #
def _euc_dist_and_rank_fun(embeddings: np.array, rankings: list, lambda_param: float) -> np.ndarray:
    """
    A tailored distance metric that uses both the euclidean distance and the rank.
    """

    dist = pairwise_distances(embeddings, embeddings)
    for i, rank in enumerate(rankings):
        dist[i, :] += rank * lambda_param
        dist[:, i] += rank * lambda_param
    return dist


# # # # Step 6  # # # #
def _minimal_augm_dist_fun(completion2embeddings: dict, completions_list: list) -> np.ndarray:
    """
    A tailored distance metric that uses the minimal euclidean distance between the
    contextual embeddings of two completions (minimal across the different paraphrases).
    """

    dist_mat = np.zeros((len(completions_list), len(completions_list)))
    for i, comp_1 in enumerate(completions_list):
        for j, comp_2 in enumerate(completions_list):
            if i == j:
                continue
            distances = []
            for embd_1 in completion2embeddings[comp_1]:
                for embd_2 in completion2embeddings[comp_2]:
                    distances.append(distance.euclidean(embd_1, embd_2))
            dist_mat[i, j] = np.min(np.array(distances)[np.array(distances) != 0])

    return dist_mat


# # # # Step 6  # # # #
def get_dbscan_clusters(embeddings_agg: dict, completions_agg: list, rankings_agg: dict, dbscan_eps_mult: float,
                        dbscan_min_samples: float, rank_weight: float, dist_func="pairwise_distances")\
                        -> [list, float]:
    """
    Returns the DBSCAN clusters with the new epsilon (might not change if converges on the first attempt).
    """

    dist_mat = _dist_func(embeddings_agg, completions_agg, rankings_agg, rank_weight=rank_weight, dist_func=dist_func)
    normalized_dist_mat = (dist_mat - np.min(dist_mat)) / (np.max(dist_mat) - np.min(dist_mat))
    clusters = DBSCAN(eps=np.abs(np.mean(normalized_dist_mat[normalized_dist_mat != 0]))*dbscan_eps_mult,
                      min_samples=dbscan_min_samples, metric="precomputed").fit_predict(normalized_dist_mat)
    # In case there are not enough clusters:
    dbscan_eps_mult_new = dbscan_eps_mult
    while len(set(clusters)) < 3:
        dbscan_eps_mult_new = dbscan_eps_mult_new + 0.05
        clusters = DBSCAN(eps=np.mean(normalized_dist_mat[normalized_dist_mat != 0]) * dbscan_eps_mult_new,
                          min_samples=dbscan_min_samples, metric="precomputed").fit_predict(normalized_dist_mat)

    return clusters, dbscan_eps_mult_new


# # # # Step 6  # # # #
def cluster_processing(clusters: list, completions_agg: list, weights_agg: dict,
                       rankings_agg: dict, rep_agg: dict, bert_top_k: int, alpha: float)\
                        -> [dict, defaultdict, defaultdict, defaultdict, defaultdict, defaultdict]:
    """
    Creates dicts that map from the cluster to its tokens, weight, rank, # of repetitions and best completion.
    """

    cluster2word_dict = {key: [] for key in set(clusters)}
    cluster2agg_ranks_dict = defaultdict(list)
    cluster2rank_dict = defaultdict(int)
    cluster2agg_weight_dict = defaultdict(list)
    cluster2weight_dict = defaultdict(int)
    cluster2agg_reps_dict = defaultdict(list)
    cluster2rep_dict = defaultdict(int)
    cluster2best_comp_dict = defaultdict()

    for i, comp in enumerate(completions_agg):
        label = clusters[i]
        cluster2word_dict[label].append(comp)
        cluster2agg_weight_dict[label].append(weights_agg[comp])
        cluster2agg_ranks_dict[label].append(rankings_agg[comp])
        cluster2agg_reps_dict[label].append(rep_agg[comp])

    for key in cluster2agg_ranks_dict.keys():
        if key == -1:
            cluster2rank_dict[key] = bert_top_k
            cluster2rep_dict[key] = 0
            cluster2weight_dict[key] = 0
        else:
            cluster2rank_dict[key] = np.min(cluster2agg_ranks_dict[key])
            cluster2rep_dict[key] = np.max(cluster2agg_reps_dict[key])
            cluster2weight_dict[key] = np.max(cluster2agg_weight_dict[key])

            weights = alpha * np.array(cluster2agg_weight_dict[key]) +\
                      (1 - alpha) * np.array(cluster2agg_reps_dict[key])
            cluster2best_comp_dict[key] = cluster2word_dict[key][np.argmax(weights)]

    return cluster2word_dict, cluster2weight_dict, cluster2rank_dict, cluster2rep_dict, cluster2best_comp_dict


# # # # Step 7  # # # #
def get_ranked_clusters(sorted_keys: list, completion2embeddings: defaultdict, cluster2word_dict: defaultdict,
                        cluster2best_comp_dict: defaultdict, top_k_completions: list):
    """
    Prints the ranked clusters and their completions and returns a dict of the sorted clusters with their tokens.
    """

    sorted_cluster2word_dict = defaultdict()

    for i, label in enumerate(sorted_keys):
        print(f"Cluster rank = {i} (label={label})")
        clust_embd_mat = np.array([completion2embeddings[comp][0] for comp in cluster2word_dict[label]])
        centroid = find_clust_centroid(clust_embd_mat, np.mean(clust_embd_mat, axis=0),
                                       cluster2word_dict[label], top_k_completions)
        try:
            b_r = top_k_completions.index(cluster2word_dict[label][centroid])
        except ValueError:
            b_r = -1
        print(f"{cluster2word_dict[label][centroid]} (BERT's rank={b_r}): {cluster2word_dict[label]}")
        sorted_cluster2word_dict[i] = (cluster2word_dict[label],
                                       cluster2word_dict[label][centroid],
                                       cluster2best_comp_dict[label])
        return sorted_cluster2word_dict


# # # # Step 7  # # # #
def find_clust_centroid(clust_embd_mat: np.array, mean_clust_embd: np.array,
                        cluster_words: defaultdict, bert_completions: list) -> int:
    """
    Returns the centroid token of each cluster using the token contextual embeddings.
    """

    sim = -1
    for embd in range(clust_embd_mat.shape[0]):
        cos_sim = np.dot(clust_embd_mat[embd, :], mean_clust_embd) / (
                np.linalg.norm(clust_embd_mat[embd, :]) * np.linalg.norm(clust_embd_mat))
        if cos_sim > sim:
            sim = cos_sim
            centroid = embd

    if cluster_words[centroid] not in bert_completions and clust_embd_mat.shape[0] > 1:
        bad_centroid = int(centroid)
        sim = -1
        for embd in range(clust_embd_mat.shape[0]):
            cos_sim = np.dot(clust_embd_mat[embd, :], mean_clust_embd) / (
                    np.linalg.norm(clust_embd_mat[embd, :]) * np.linalg.norm(clust_embd_mat))
            if cos_sim > sim and embd != bad_centroid:
                sim = cos_sim
                centroid = embd

    return centroid
