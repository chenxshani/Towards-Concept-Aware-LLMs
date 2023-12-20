import argparse
import openai
import pickle
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import time
import os
from os import listdir
from os.path import isfile, join


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="text-davinci-003",
        type=str,
        required=False,
        help="Which GPT model to use."
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

    openai_key = r""

    with open("100-everyday-things-hypernym-up.pickle", "rb") as pf:
        hypernym = pickle.load(pf)
    with open("100-everyday-things-hyponym-down.pickle", "rb") as pf:
        hyponym = pickle.load(pf)

    et_hypernym_counter = defaultdict(int)
    et_hyponym_counter = defaultdict(int)

    asymmetry = args.asymmetry
    transitivity = args.transitivity
    property_inher = args.property_inher

    if asymmetry:
        et_dumped = {f.split("-")[3] for f in listdir("gpt-3.5-asym-dump\\") if isfile(join("gpt-3.5-asym-dump\\", f))}
    else:
        et_dumped = {f.split("-")[3] for f in listdir("gpt-3.5-dump\\") if isfile(join("gpt-3.5-dump\\", f))}

    if not transitivity and not property_inher:

        for et in tqdm(list(hypernym.keys())):
            if et in et_dumped:
                continue
            et_hypernym_counter_tmp = 0
            et_hyponym_counter_tmp = 0
            if len(hypernym[et]) > 0:
                for et_hypernym in hypernym[et]:
                    time.sleep(10)

                    gpt_response = openai.Completion.create(
                        api_key=openai_key,
                        model=args.model_name,
                        prompt=f"Is {et} a type of {et_hypernym}?",
                        max_tokens=5
                    )

                    if asymmetry:
                        gpt_response_asymmetry = openai.Completion.create(
                                                                api_key=openai_key,
                                                                model=args.model_name,
                                                                prompt=f"Is {et_hypernym} a type of {et}?",  # Asymmetry
                                                                max_tokens=5
                                                                )


                    time.sleep(10)
                    print(f"Is {et} a type of {et_hypernym}?")
                    print(gpt_response["choices"][0]["text"])

                    if asymmetry:
                        if ("yes" or "true") in gpt_response["choices"][0]["text"].lower():  # Asymmetry
                            et_hypernym_counter_tmp += int(("yes" or "true")
                                                           in gpt_response_asymmetry["choices"][0]["text"].lower())  # Asymmetry
                    else:
                        et_hypernym_counter_tmp += int(("yes" or "true") in gpt_response["choices"][0]["text"].lower())
                et_hypernym_counter[et] = et_hypernym_counter_tmp / len(hypernym[et])
            if len(hyponym[et]) > 0:
                for et_hyponym in hyponym[et]:
                    time.sleep(10)

                    gpt_response = openai.Completion.create(
                        api_key=openai_key,
                        model=args.model_name,
                        prompt=f"Is {et_hyponym} a type of {et}?",
                        max_tokens=5
                    )

                    if asymmetry:
                        gpt_response_asymmetry = openai.Completion.create(
                                                                api_key=openai_key,
                                                                model=args.model_name,
                                                                prompt=f"Is {et} a type of {et_hyponym}?",  # Asymmetry
                                                                max_tokens=5
                                                                )

                    time.sleep(10)
                    print(f"Is {et_hyponym} a type of {et}?")
                    print(gpt_response["choices"][0]["text"])
                    if asymmetry:
                        if ("yes" or "true") in gpt_response["choices"][0]["text"].lower():  # Asymmetry
                            et_hyponym_counter_tmp += int(("yes" or "true") in gpt_response_asymmetry["choices"][0]["text"].lower())  # Asymmetry
                    else:
                        et_hyponym_counter_tmp += int(("yes" or "true") in gpt_response["choices"][0]["text"].lower())
                et_hyponym_counter[et] = et_hyponym_counter_tmp / len(hyponym[et])

            if asymmetry:
                with open(f"gpt-3.5-asym-dump\\100-everyday-things-{et}-hypernym-gpt.pickle", "wb") as pf:
                    pickle.dump(et_hypernym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"gpt-3.5-asym-dump\\100-everyday-things-{et}-hyponym-gpt.pickle", "wb") as pf:
                    pickle.dump(et_hyponym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(f"gpt-3.5-dump\\100-everyday-things-{et}-hypernym-gpt.pickle", "wb") as pf:
                    pickle.dump(et_hypernym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"gpt-3.5-dump\\100-everyday-things-{et}-hyponym-gpt.pickle", "wb") as pf:
                    pickle.dump(et_hyponym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)

        if asymmetry:
            folder = "gpt-3.5-asym-dump\\"
        else:
            folder = "gpt-3.5-dump\\"
        et_hypernym_counter = {}
        for filename in os.listdir(folder):
            if filename.endswith('-hypernym-gpt.pickle'):
                myfile = open(folder + filename, "rb")
                et_hypernym_counter[os.path.splitext(filename)[0].split("-")[3]] = pickle.load(myfile)
                myfile.close()
                print(filename)

        et_hyponym_counter = {}
        for filename in os.listdir(folder):
            if filename.endswith('-hyponym-gpt.pickle'):
                myfile = open(folder + filename, "rb")
                et_hyponym_counter[os.path.splitext(filename)[0].split("-")[3]] = pickle.load(myfile)
                myfile.close()
                print(filename)

        with open("100-everyday-things-hypernym-gpt.pickle", "wb") as pf:
            pickle.dump(et_hypernym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)
        with open("100-everyday-things-hyponym-gpt.pickle", "wb") as pf:
            pickle.dump(et_hyponym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)




        et_hypernym_counter = [et_hypernym_counter[key][key] for key in et_hypernym_counter.keys()]
        print(np.mean(et_hypernym_counter))

        et_hyponym_counter = [et_hyponym_counter[key][key] for key in et_hyponym_counter.keys()]
        print(np.mean(et_hyponym_counter))

    else:
        if transitivity:
            i = 0
            transitivity_dict = defaultdict(int)
            for et in tqdm({f.split("-")[3] for f in listdir("gpt-3.5-dump\\") if isfile(join("gpt-3.5-dump\\", f))}):
                et_tmp_trans = []
                if len(hypernym[et]) > 0:
                    for et_hypernym in hypernym[et]:
                        time.sleep(1)
                        if ("yes" or "true") in openai.Completion.create(
                                api_key=openai_key,
                                model=args.model_name,
                                prompt=f"Is {et} a type of {et_hypernym}?",
                                max_tokens=5
                        )["choices"][0]["text"].lower():
                            if len(hyponym[et]) > 0:
                                for et_hyponym in hyponym[et]:
                                    time.sleep(1)
                                    if ("yes" or "true") in openai.Completion.create(
                                            api_key=openai_key,
                                            model=args.model_name,
                                            prompt=f"Is {et_hyponym} a type of {et}?",
                                            max_tokens=5
                                    )["choices"][0]["text"].lower():
                                        i += 1
                                        print(et)
                                        gpt_response = openai.Completion.create(
                                            api_key=openai_key,
                                            model=args.model_name,
                                            prompt=f"Is {et_hyponym} a type of {et_hypernym}?",
                                            max_tokens=5
                                        )
                                        print(f"Is {et_hyponym} a type of {et_hypernym}?")
                                        print(gpt_response["choices"][0]["text"])
                                        et_tmp_trans.append(
                                            int(("yes" or "true") in gpt_response["choices"][0]["text"].lower()))
                                        time.sleep(1)
                                        et_tmp_trans.append(
                                            int(("yes" or "true") in gpt_response["choices"][0]["text"].lower()))
                if len(et_tmp_trans) > 0:
                    transitivity_dict[et] = np.mean(et_tmp_trans)

            print(i)
            print(len(transitivity_dict))
            print(np.mean(list(transitivity_dict.values())))

        elif property_inher:

            with open("quasimodo_et2q.pickle", "rb") as pf:
                et2q = pickle.load(pf)

            et_inher_dict = defaultdict(int)
            hypo_inher_dict = defaultdict(int)
            hypo_cond_et_dict = defaultdict(int)
            e, h, hge = 0, 0, 0

            for et in tqdm(list(hypernym.keys())):
                et_tmp_inher, hypo_tmp_inher, hypo_cond_et_tmp_inher = [], [], []
                for proprt in et2q[et]:
                    if ("yes" or "true") in openai.Completion.create(
                            api_key=openai_key,
                            model=args.model_name,
                            prompt=proprt["masked sentence"],
                            max_tokens=5)["choices"][0]["text"].lower():
                        time.sleep(1)
                        et_tmp_inher.append(int(
                            ("yes" or "true") in openai.Completion.create(
                                api_key=openai_key,
                                model=args.model_name,
                                prompt=proprt["masked sentence"].replace(proprt["hyper"], et),
                                max_tokens=5)["choices"][0]["text"].lower()))
                        time.sleep(1)
                        for et_hyponym in proprt["hyponyms"]:
                            resp = ("yes" or "true") in openai.Completion.create(
                                api_key=openai_key,
                                model=args.model_name,
                                prompt=proprt["masked sentence"].replace(proprt["hyper"], et_hyponym),
                                max_tokens=5)["choices"][0]["text"].lower()
                            time.sleep(1)
                            hypo_tmp_inher.append(int(resp))
                            if et_tmp_inher[-1] == 1:
                                hypo_cond_et_tmp_inher.append(int(resp))
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
