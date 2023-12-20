# # # IMPORTS # # #
import argparse
from utilities.bertMaskPred import *
import pickle
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import transformers
from transformers import BertTokenizer, BertForMaskedLM



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
        default=50,
        type=int,
        required=False,
        help="How many token completions from the top of BERT's ranked "
             "vocabulary to use for the concept-aware manipulation.",
    )

    parser.add_argument(
        "--asymmetry",
        default=False,
        action="store_true",
        required=False,
        help="If True the RQ2 asymmetry analysis is performed."
    )

    parser.add_argument(
        "--transitivity",
        default=False,
        action="store_true",
        required=False,
        help="If True the RQ2 transitivity analysis is performed."
    )

    parser.add_argument(
        "--property_inher",
        default=False,
        action="store_true",
        required=False,
        help="If True the RQ2 property inheritance analysis is performed."
    )

    return parser.parse_args()


def main(args):

    with open("100-everyday-things-hypernym-up.pickle", "rb") as pf:
        hypernym = pickle.load(pf)
    with open("100-everyday-things-hyponym-down.pickle", "rb") as pf:
        hyponym = pickle.load(pf)

    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForMaskedLM.from_pretrained(args.model_name)
    model.eval()

    et_hypernym_counter_k_list = defaultdict(list)
    et_hyponym_counter_k_list = defaultdict(list)

    model_top_k = args.bert_top_k
    k_values = np.arange(1, model_top_k)

    asymmetry = args.asymmetry
    transitivity = args.transitivity
    property_inher = args.property_inher

    if not transitivity and not property_inher:

        for et in tqdm(list(hypernym.keys())):
            et_hypernym_counter = defaultdict(list)
            if len(hypernym[et]) > 0:
                for et_hypernym in hypernym[et]:
                    top_k_tokens = predict_masked_sent(
                        f"{et} is a type of [MASK].", model, tokenizer, top_k=model_top_k)
                    if asymmetry:
                        top_k_tokens_asymmetry = predict_masked_sent(
                            f"{et_hypernym} is a type of [MASK].", model, tokenizer, top_k=model_top_k)  # Asymmetry
                        for k in k_values:  # Asymmetry
                            if et in top_k_tokens[:k]:  # Asymmetry
                                et_hypernym_counter[k].append(int(et in top_k_tokens_asymmetry[:k]))  # Asymmetry
                    else:
                        for k in k_values:
                            et_hypernym_counter[k].append(int(et_hypernym in top_k_tokens[:k]))
                if asymmetry:
                    for k in k_values:  # Asymmetry
                        et_hypernym_counter_k_list[k].append(np.nanmean(et_hypernym_counter[k]))  # Asymmetry
                else:
                    for k in k_values:
                        et_hypernym_counter_k_list[k].append(np.nanmean(et_hypernym_counter[k]))
            et_hyponym_counter = defaultdict(list)
            if len(hyponym[et]) > 0:
                for et_hyponym in hyponym[et]:
                    top_k_tokens = predict_masked_sent(
                        f"{et_hyponym} is a type of [MASK].", model, tokenizer, top_k=model_top_k)
                    if asymmetry:
                        top_k_tokens_asymmetry = predict_masked_sent(
                            f"{et} is a type of [MASK].", model, tokenizer, top_k=model_top_k)  # Asymmetry
                        for k in k_values:
                            if et in top_k_tokens[:k]:  # Asymmetry
                                et_hyponym_counter[k].append(int(et_hyponym in top_k_tokens_asymmetry[:k]))  # Asymmetry
                    else:
                        for k in k_values:
                            et_hyponym_counter[k].append(int(et in top_k_tokens[:k]))
                if asymmetry:
                    for k in k_values:  # Asymmetry
                        et_hyponym_counter_k_list[k].append(np.nanmean(et_hyponym_counter[k]))  # Asymmetry
                else:
                    for k in k_values:
                        et_hyponym_counter_k_list[k].append(np.nanmean(et_hyponym_counter[k]))

        et_hypernym_counter_mean = [np.nanmean(list(et_hypernym_counter_k_list[k])) for k in k_values]
        et_hyponym_counter_mean = [np.nanmean(list(et_hyponym_counter_k_list[k])) for k in k_values]
        comb_counter_mean = [np.nanmean(et_hyponym_counter_k_list[k] + et_hypernym_counter_k_list[k]) for k in k_values]
        plt.plot(k_values, et_hypernym_counter_mean, label="Hypernym", color="r", markersize=7)
        plt.plot(k_values, et_hyponym_counter_mean, label="Hyponym", color="b", markersize=7)
        plt.plot(k_values, comb_counter_mean, "--g", label="Combined", markersize=7)
        plt.xlabel("K", fontsize=18)
        plt.ylabel("Mean Retrival", fontsize=18)
        plt.legend()
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        if asymmetry:
            print("Asymmetry")
            plt.title("Mean Conceptual Retrival of BERT's Asymmetry\n Property for 100 Everyday Things as a Function of K",
                      fontsize=16)  # Asymmetry
            plt.savefig("BERT_asymmetry.png", dpi=300)  # Asymmetry
        else:
            plt.title("Mean Conceptual Retrival of BERT for\n100 Everyday Things as a Function of K", fontsize=18)
            plt.savefig("BERT.png", dpi=300)
        plt.show()

        print(et_hypernym_counter_mean)
        print(et_hyponym_counter_mean)

        et_hypernym_counter_max = [np.max(list(et_hypernym_counter_k_list[k])) for k in k_values]
        et_hyponym_counter_max = [np.max(list(et_hyponym_counter_k_list[k])) for k in k_values]


        plt.plot(k_values, et_hypernym_counter_max, label="Hypernym", color="r")
        plt.plot(k_values, et_hyponym_counter_max, label="Hyponym", color="b")
        plt.xlabel("K")
        plt.ylabel("Max Retrival")
        plt.title("Max Conceptual Retrival of BERT for\n100 Everyday Things as a Function of K")
        plt.legend()
        plt.show()

    else:
        if transitivity:
            i = 0
            for et in tqdm(list(hypernym.keys())):
                et_hypernym_counter = defaultdict(list)
                if len(hypernym[et]) > 0:
                    for et_hypernym in hypernym[et]:
                        top_k_tokens = predict_masked_sent(f"{et} is a type of [MASK].",
                                                           model, tokenizer, top_k=model_top_k)
                        if et_hypernym in top_k_tokens:
                            if len(hyponym[et]) > 0:
                                for et_hyponym in hyponym[et]:
                                    top_k_tokens = predict_masked_sent(f"{et_hyponym} is a type of [MASK].",
                                                                       model, tokenizer, top_k=model_top_k)
                                    if et in top_k_tokens:
                                        i += 1
                                        top_k_tokens = predict_masked_sent(f"{et_hyponym} is a type of [MASK].",
                                                                           model, tokenizer, top_k=model_top_k)
                                        for k in k_values:
                                            et_hypernym_counter[k].append(int(et_hypernym in top_k_tokens[:k]))

                                for k in k_values:
                                    et_hypernym_counter_k_list[k].append(np.nanmean(et_hypernym_counter[k]))

            print(i)
            et_hypernym_counter_mean = [np.nanmean(list(et_hypernym_counter_k_list[k])) for k in k_values]
            print(et_hypernym_counter_mean)
            plt.plot(k_values, et_hypernym_counter_mean, label="Hypernym", color="r", markersize=7)
            plt.xlabel("K", fontsize=18)
            plt.ylabel("Mean Retrival", fontsize=18)
            plt.title(
                "Mean Conceptual Retrival of BERT's Transitivity\n Property for 100 Everyday Things as a Function of K",
                fontsize=16)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.savefig("BERT_trans.png", dpi=300)
            plt.show()

        elif property_inher:
            with open("quasimodo_et2transitivity_dict.pickle", "rb") as pf:
                et2sen = pickle.load(pf)

            et_inher_dict = defaultdict(int)
            hypo_inher_dict = defaultdict(int)
            hypo_cond_et_dict = defaultdict(int)
            e, h, hge = 0, 0, 0

            for et in tqdm(list(hypernym.keys())):
                et_tmp_inher, hypo_tmp_inher, hypo_cond_et_tmp_inher = [], [], []
                for proprt in et2sen[et]:
                    top_k_tokens = predict_masked_sent(proprt["masked sentence"].replace(et, proprt["hyper"]),
                                                       model, tokenizer, top_k=model_top_k)
                    if proprt["object"] in top_k_tokens:
                        top_k_tokens = predict_masked_sent(proprt["masked sentence"], model, tokenizer,
                                                           top_k=model_top_k)
                        et_tmp_inher.append(int(proprt["object"] in top_k_tokens))
                        for et_hyponym in proprt["hyponyms"]:
                            top_k_tokens = predict_masked_sent(proprt["masked sentence"].replace(et, et_hyponym),
                                                               model, tokenizer, top_k=model_top_k)
                            hypo_tmp_inher.append(int(proprt["object"] in top_k_tokens))
                            if et_tmp_inher[-1] == 1:
                                hypo_cond_et_tmp_inher.append(int(proprt["object"] in top_k_tokens))
                if len(et_tmp_inher) > 0:
                    et_inher_dict[et] = np.mean(et_tmp_inher)
                    e += len(et_tmp_inher)
                if len(hypo_tmp_inher) > 0:
                    hypo_inher_dict[et] = np.mean(hypo_tmp_inher)
                    h += len(hypo_tmp_inher)
                if len(hypo_cond_et_tmp_inher) > 0:
                    hypo_cond_et_dict[et] = np.mean(hypo_cond_et_tmp_inher)
                    hge += len(hypo_cond_et_tmp_inher)

            print(e, h, hge)

            print(len(et_inher_dict.keys()))
            print(len(hypo_inher_dict.keys()))
            print(len(hypo_cond_et_dict.keys()))

            print(np.mean(list(et_inher_dict.values())))
            print(np.mean(list(hypo_inher_dict.values())))
            print(np.mean(list(hypo_cond_et_dict.values())))

if __name__ == "__main__":
    main(parse_args())
