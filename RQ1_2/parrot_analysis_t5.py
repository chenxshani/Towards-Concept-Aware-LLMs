import argparse
from utilities.t5MaskPred import *
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="t5-large",
        type=str,
        required=False,
        help="Which T5 model to use."
    )
    parser.add_argument(
        "--t5_top_k",
        default=50,
        type=int,
        required=False,
        help="How many token completions from the top of T5's ranked "
             "vocabulary to retrieve.",
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

    BEAMS_MULTP, N_SEN_RETURN, MAX_LEN = 2, 1, 20

    with open("100-everyday-things-hypernym-up.pickle", "rb") as pf:
        hypernym = pickle.load(pf)
    with open("100-everyday-things-hyponym-down.pickle", "rb") as pf:
        hyponym = pickle.load(pf)

    t5 = T5(t5_path=args.model_name)

    et_hypernym_counter_k_list = defaultdict(list)
    et_hyponym_counter_k_list = defaultdict(list)

    model_top_k = args.t5_top_k
    k_values = np.arange(1, model_top_k)

    asymmetry = args.asymmetry
    transitivity = args.transitivity
    property_inher = args.property_inher


    if not transitivity and not property_inher:

        for et in tqdm(list(hypernym.keys())):
            et_hypernym_counter = defaultdict(list)
            if len(hypernym[et]) > 0:
                for et_hypernym in hypernym[et]:
                    top_k_tokens = t5.get_top_k_predictions(f"{et} is a type of <extra_id_0>.",
                                                                              topk=model_top_k,
                                                                              beams_multip=BEAMS_MULTP,
                                                                              max_len=MAX_LEN)
                    if asymmetry:
                        # Asymmetry
                        top_k_tokens_asymmetry = t5.get_top_k_predictions(f"{et_hypernym} is a type of <extra_id_0>.",
                                                                topk=model_top_k,
                                                                beams_multip=BEAMS_MULTP,
                                                                max_len=MAX_LEN)  # Asymmetry
                    for k in k_values:
                        if asymmetry:
                            if et_hypernym in top_k_tokens[0][:k]:  # Asymmetry
                                et_hypernym_counter[k].append(int(et in top_k_tokens_asymmetry[0][:k]))  # Asymmetry
                        else:
                            et_hypernym_counter[k].append(int(et_hypernym in top_k_tokens[0][:k]))
                for k in k_values:
                    et_hypernym_counter_k_list[k].append(np.nanmean(et_hypernym_counter[k]))

            et_hyponym_counter = defaultdict(list)
            if len(hyponym[et]) > 0:
                for et_hyponym in hyponym[et]:
                    top_k_tokens = t5.get_top_k_predictions(f"{et_hyponym} is a type of <extra_id_0>.",
                                                                              topk=model_top_k,
                                                                              beams_multip=BEAMS_MULTP,
                                                                              max_len=MAX_LEN)
                    if asymmetry:
                        # Asymmetry
                        top_k_tokens_asymmetry = t5.get_top_k_predictions(f"{et} is a type of <extra_id_0>.",
                                                                                  topk=model_top_k,
                                                                                  beams_multip=BEAMS_MULTP,
                                                                                  max_len=MAX_LEN)  # Asymmetry
                    for k in k_values:
                        if asymmetry:
                            if et in top_k_tokens[0][:k]:  # Asymmetry
                                et_hyponym_counter[k].append(int(et_hyponym in top_k_tokens_asymmetry[0][:k]))  # Asymmetry
                        else:
                            et_hyponym_counter[k].append(int(et in top_k_tokens[0][:k]))
                for k in k_values:
                    et_hyponym_counter_k_list[k].append(np.nanmean(et_hyponym_counter[k]))

        et_hypernym_counter_mean = [np.nanmean(list(et_hypernym_counter_k_list[k])) for k in k_values]
        et_hyponym_counter_mean = [np.nanmean(list(et_hyponym_counter_k_list[k])) for k in k_values]
        comb_counter_mean = [np.nanmean(et_hyponym_counter_k_list[k] + et_hypernym_counter_k_list[k]) for k in k_values]
        plt.plot(k_values, et_hypernym_counter_mean, label="Hypernym", color="r")
        plt.plot(k_values, et_hyponym_counter_mean, label="Hyponym", color="b")
        plt.plot(k_values, comb_counter_mean, "--g", label="Combined", markersize=7)
        plt.xlabel("K", fontsize=18)
        plt.ylabel("Mean Retrival", fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend()
        if asymmetry:
            plt.title(f"Mean Conceptual Retrival of T5's Asymmetry\n"
                      f"Property for 100 Everyday Things as a Function of K", fontsize=16)  # Asymmetry
            plt.savefig("T5_asymmetry.png", dpi=300)  # Asymmetry
        else:
            plt.title(f"Mean Conceptual Retrival of T5 for\n100 Everyday Things as a Function of K", fontsize=16)
            plt.savefig("T5.png", dpi=300)
        plt.show()
        if asymmetry:
            print("Asymmetry")
        print(et_hypernym_counter_mean)
        print(et_hyponym_counter_mean)

        et_hypernym_counter_max = [np.max(list(et_hypernym_counter_k_list[k])) for k in k_values]
        et_hyponym_counter_max = [np.max(list(et_hyponym_counter_k_list[k])) for k in k_values]


        plt.plot(k_values, et_hypernym_counter_max, label="Hypernym", color="r")
        plt.plot(k_values, et_hyponym_counter_max, label="Hyponym", color="b")
        plt.xlabel("K")
        plt.ylabel("Max Retrival")
        plt.legend()
        if asymmetry:
            plt.title(f"Max Conceptual Retrival of T5's Asymmetry\n"
                      f"Property for 100 Everyday Things as a Function of K")  # Asymmetry
            plt.savefig("T5_max_asymmetry.png", dpi=300)  # Asymmetry
        else:
            plt.title(f"Max Conceptual Accuracy of T5 for\n100 Everyday Things as a Function of K")
            plt.savefig("T5_max.png", dpi=300)
        plt.show()

    else:
        if transitivity:
            i = 0

            for et in tqdm(list(hypernym.keys())):
                et_hypernym_counter = defaultdict(list)
                if len(hypernym[et]) > 0:
                    for et_hypernym in hypernym[et]:
                        top_k_tokens = t5.get_top_k_predictions(f"{et} is a type of <extra_id_0>.",
                                                                topk=model_top_k,
                                                                beams_multip=BEAMS_MULTP,
                                                                max_len=MAX_LEN)
                        if et_hypernym in top_k_tokens[0]:
                            if len(hyponym[et]) > 0:
                                for et_hyponym in hyponym[et]:
                                    top_k_tokens = t5.get_top_k_predictions(f"{et_hyponym} is a type of <extra_id_0>.",
                                                                            topk=model_top_k,
                                                                            beams_multip=BEAMS_MULTP,
                                                                            max_len=MAX_LEN)
                                    if et in top_k_tokens[0]:
                                        i += 1
                                        top_k_tokens = t5.get_top_k_predictions(
                                            f"{et_hyponym} is a type of <extra_id_0>.",
                                            topk=model_top_k,
                                            beams_multip=BEAMS_MULTP,
                                            max_len=MAX_LEN)
                                        for k in k_values:
                                            et_hypernym_counter[k].append(int(et_hypernym in top_k_tokens[0][:k]))

                                for k in k_values:
                                    et_hypernym_counter_k_list[k].append(np.nanmean(et_hypernym_counter[k]))

            print(i)
            et_hypernym_counter_mean = [np.nanmean(list(et_hypernym_counter_k_list[k])) for k in k_values]
            print(et_hypernym_counter_mean)
            plt.plot(k_values, et_hypernym_counter_mean, label="Hypernym", color="r", markersize=7)
            plt.xlabel("K", fontsize=18)
            plt.ylabel("Mean Retrival", fontsize=18)
            plt.title(
                "Mean Conceptual Retrival of T5's Transitivity\n Property for 100 Everyday Things as a Function of K",
                fontsize=16)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.savefig("T5_trans.png", dpi=300)
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
                        try:
                            top_k_tokens = t5.get_top_k_predictions(
                                proprt["masked sentence"].replace(et, proprt["hyper"]).replace("[MASK]",
                                                                                               "<extra_id_0>"),
                                topk=model_top_k,
                                beams_multip=BEAMS_MULTP,
                                max_len=MAX_LEN)[0]
                            if proprt["object"] in top_k_tokens:
                                top_k_tokens = \
                                t5.get_top_k_predictions(proprt["masked sentence"].replace("[MASK]", "<extra_id_0>"),
                                                         topk=model_top_k,
                                                         beams_multip=BEAMS_MULTP,
                                                         max_len=MAX_LEN)[0]
                                et_tmp_inher.append(int(proprt["object"] in top_k_tokens))
                                for et_hyponym in proprt["hyponyms"]:
                                    top_k_tokens = t5.get_top_k_predictions(
                                        proprt["masked sentence"].replace(et, et_hyponym).replace("[MASK]",
                                                                                                  "<extra_id_0>"),
                                        topk=model_top_k,
                                        beams_multip=BEAMS_MULTP,
                                        max_len=MAX_LEN)[0]
                                    hypo_tmp_inher.append(int(proprt["object"] in top_k_tokens))
                                    if et_tmp_inher[-1] == 1:
                                        hypo_cond_et_tmp_inher.append(int(proprt["object"] in top_k_tokens))
                        except:
                            continue
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


