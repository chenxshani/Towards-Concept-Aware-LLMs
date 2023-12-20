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
        default="gpt-4",
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
        et_dumped = {f.split("-")[3] for f in listdir("gpt4-asym-dump\\") if isfile(join("gpt4-asym-dump\\", f))}
    else:
        et_dumped = {f.split("-")[3] for f in listdir("gpt4-dump\\") if isfile(join("gpt4-dump\\", f))}


    if not transitivity and not property_inher:

        for et in tqdm(list(hypernym.keys())):
            if et in et_dumped:
                continue
            et_hypernym_counter_tmp = 0
            et_hyponym_counter_tmp = 0
            if len(hypernym[et]) > 0:
                for et_hypernym in hypernym[et]:

                    gpt_response = openai.ChatCompletion.create(
                                                            api_key=openai_key,
                                                            model=args.model_name,
                                                            messages=[{"role": "user",
                                                                       "content": f"Is {et} a type of {et_hypernym}?"}],
                                                            max_tokens=5
                                                            )

                    if asymmetry:
                        gpt_response_asymmetry = openai.ChatCompletion.create(
                            api_key=openai_key,
                            model=args.model_name,
                            messages=[{"role": "user",
                                       "content": f"Is {et_hypernym} a type of {et}?"}],
                            max_tokens=5
                        )

                    print(f"Is {et} a type of {et_hypernym}?")
                    time.sleep(15)
                    print(gpt_response["choices"][0]["message"]["content"])
                    if not asymmetry:
                        et_hypernym_counter_tmp += int(("yes" or "true") in gpt_response["choices"][0]["message"]["content"].lower())
                    else:
                        if ("yes" or "true") in gpt_response["choices"][0]["message"]["content"].lower():  # Asymmetry
                            et_hypernym_counter_tmp += int(("yes" or "true") in gpt_response_asymmetry["choices"][0]["message"]["content"].lower())
                et_hypernym_counter[et] = et_hypernym_counter_tmp / len(hypernym[et])
            if len(hyponym[et]) > 0:
                for et_hyponym in hyponym[et]:

                    gpt_response = openai.ChatCompletion.create(
                                                            api_key=openai_key,
                                                            model=args.model_name,
                                                            messages=[{"role": "user",
                                                                       "content": f"Is {et_hyponym} a type of {et}?"}],
                                                            max_tokens=5
                                                            )

                    if asymmetry:
                        gpt_response_asymmetry = openai.ChatCompletion.create(
                            api_key=openai_key,
                            model=args.model_name,
                            messages=[{"role": "user",
                                       "content": f"Is {et} a type of {et_hyponym}?"}],
                            max_tokens=5
                        )
                    time.sleep(15)
                    print(f"Is {et_hyponym} a type of {et}?")
                    print(gpt_response["choices"][0]["message"]["content"])
                    if not asymmetry:
                        et_hyponym_counter_tmp += int(("yes" or "true") in gpt_response["choices"][0]["message"]["content"].lower())
                    else:
                        if ("yes" or "true") in gpt_response["choices"][0]["message"]["content"].lower():  # Asymmetry
                            et_hyponym_counter_tmp += int(("yes" or "true") in gpt_response_asymmetry["choices"][0]["message"]["content"].lower())
                et_hyponym_counter[et] = et_hyponym_counter_tmp / len(hyponym[et])
            if asymmetry:
                with open(f"gpt4-asym-dump\\100-everyday-things-{et}-hypernym-gpt4.pickle", "wb") as pf:
                    pickle.dump(et_hypernym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"gpt4-asym-dump\\100-everyday-things-{et}-hyponym-gpt4.pickle", "wb") as pf:
                    pickle.dump(et_hyponym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(f"gpt4-dump\\100-everyday-things-{et}-hypernym-gpt4.pickle", "wb") as pf:
                    pickle.dump(et_hypernym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)
                with open(f"gpt4-dump\\100-everyday-things-{et}-hyponym-gpt4.pickle", "wb") as pf:
                    pickle.dump(et_hyponym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)

        if asymmetry:
            folder = "gpt4-asym-dump\\"
        else:
            folder = "gpt4-dump\\"
        et_hypernym_counter = {}
        for filename in os.listdir(folder):
            if filename.endswith('-hypernym-gpt4.pickle'):
                myfile = open(folder + filename, "rb")
                et_hypernym_counter[os.path.splitext(filename)[0].split("-")[3]] = pickle.load(myfile)
                myfile.close()
                print(filename)

        et_hyponym_counter = {}
        for filename in os.listdir(folder):
            if filename.endswith('-hyponym-gpt4.pickle'):
                myfile = open(folder + filename, "rb")
                et_hyponym_counter[os.path.splitext(filename)[0].split("-")[3]] = pickle.load(myfile)
                myfile.close()
                print(filename)

        with open("100-everyday-things-hypernym-gpt4.pickle", "wb") as pf:
            pickle.dump(et_hypernym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)
        with open("100-everyday-things-hyponym-gpt4.pickle", "wb") as pf:
            pickle.dump(et_hyponym_counter, pf, protocol=pickle.HIGHEST_PROTOCOL)

        et_hypernym_counter = [et_hypernym_counter[key][key] for key in et_hypernym_counter.keys()]
        if asymmetry:
            print(1-np.mean(et_hypernym_counter))
        else:
            print(np.mean(et_hypernym_counter))

        et_hyponym_counter = [et_hyponym_counter[key][key] for key in et_hyponym_counter.keys()]
        if asymmetry:
            print(1-np.mean(et_hyponym_counter))
        else:
            print(np.mean(et_hyponym_counter))

    else:
        if transitivity:

            i = 0
            transitivity_dict = defaultdict(int)
            for et in tqdm({f.split("-")[3] for f in listdir("gpt-dump\\") if isfile(join("gpt-dump\\", f))}):
                et_tmp_trans = []
                if len(hypernym[et]) > 0:
                    for et_hypernym in hypernym[et]:
                        time.sleep(1)
                        if ("yes" or "true") in openai.ChatCompletion.create(
                                    api_key=openai_key,
                                    model=args.model_name,
                                    messages=[{"role": "user", "content": f"Is {et} a type of {et_hypernym}?"}],
                                    max_tokens=5)["choices"][0]["message"]["content"].lower():
                            if len(hyponym[et]) > 0:
                                for et_hyponym in hyponym[et]:
                                    time.sleep(1)
                                    if ("yes" or "true") in openai.ChatCompletion.create(
                                    api_key=openai_key,
                                    model=args.model_name,
                                    messages=[{"role": "user", "content": f"Is {et_hyponym} a type of {et}?"}],
                                    max_tokens=5)["choices"][0]["message"]["content"].lower():
                                        i += 1
                                        print(et)
                                        gpt_response = openai.ChatCompletion.create(
                                                        api_key=openai_key,
                                                        model=args.model_name,
                                                        messages=[{"role": "user",
                                                                   "content": f"Is {et_hyponym} a type of {et_hypernym}?"}],
                                                        max_tokens=5)
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
            ii = 0

            et_resp_dict = defaultdict()
            et_tmp_inher_dict = defaultdict(list)
            hypo_tmp_inher_dict = defaultdict(list)

            et_dumped = {f.split("-")[3] for f in listdir("gpt4-dump-inher\\") if
                         isfile(join("gpt4-dump-inher\\", f)) and f.endswith("pickle")}

            for et in tqdm(list(hypernym.keys())):
                if et in et_dumped:
                    print(f"{et} already saved")
                    continue
                print(et)
                et_tmp_inher, hypo_tmp_inher, hypo_cond_et_tmp_inher = [], [], []
                for prompt in et2q[et]:
                    ii += 1
                    print(ii, 0)
                    et_resp = openai.ChatCompletion.create(
                        api_key=openai_key,
                        model=args.model_name,
                        messages=[{"role": "user", "content": prompt["masked sentence"]}],
                        max_tokens=5)
                    et_resp_dict[f"{et}: {prompt['masked sentence']}"] = et_resp["choices"][0]["message"][
                        "content"].lower()
                    with open(f"gpt4-dump-inher\\100-everyday-things-{et}-et_resp_dict-gpt4.pickle", "wb") as pf:
                        pickle.dump(et_resp_dict, pf, protocol=pickle.HIGHEST_PROTOCOL)
                    time.sleep(60)

                    if ("yes" or "true") in et_resp["choices"][0]["message"]["content"].lower():
                        ii += 1
                        print(ii, 1)
                        resp = openai.ChatCompletion.create(
                            api_key=openai_key,
                            model=args.model_name,
                            messages=[
                                {"role": "user", "content": prompt["masked sentence"].replace(prompt["hyper"], et)}],
                            max_tokens=5)
                        et_tmp_inher.append(int(("yes" or "true") in resp["choices"][0]["message"]["content"].lower()))
                        et_tmp_inher_dict[f"{et}: {prompt['masked sentence']}"].append(
                            resp["choices"][0]["message"]["content"].lower())
                        with open(f"gpt4-dump-inher\\100-everyday-things-{et}-et_tmp_inher_dict-gpt4.pickle",
                                  "wb") as pf:
                            pickle.dump(et_tmp_inher_dict, pf, protocol=pickle.HIGHEST_PROTOCOL)
                        time.sleep(60)

                        for et_hyponym in prompt["hyponyms"]:
                            ii += 1
                            print(ii, 2)
                            resp_hypo = openai.ChatCompletion.create(
                                api_key=openai_key,
                                model=args.model_name,
                                messages=[{"role": "user",
                                           "content": prompt["masked sentence"].replace(prompt["hyper"], et_hyponym)}],
                                max_tokens=5)
                            binary_resp_hypo = ("yes" or "true") in resp_hypo["choices"][0]["message"][
                                "content"].lower()
                            hypo_tmp_inher_dict[f"{et}: {prompt['masked sentence']}: {et_hyponym}"].append(
                                resp_hypo["choices"][0]["message"]["content"].lower())
                            with open(
                                    f"gpt4-dump-inher\\100-everyday-things-{et}-{et_hyponym}-hypo_tmp_inher_dict-gpt4.pickle",
                                    "wb") as pf:
                                pickle.dump(hypo_tmp_inher_dict, pf, protocol=pickle.HIGHEST_PROTOCOL)
                            time.sleep(60)

                            hypo_tmp_inher.append(int(binary_resp_hypo))
                            if et_tmp_inher[-1] == 1:
                                hypo_cond_et_tmp_inher.append(int(binary_resp_hypo))

                if len(et2q[et]) > 0:
                    print("saving")
                    with open(f"gpt4-dump-inher\\100-everyday-things-{et}-hypernym-et_tmp_inher-gpt4.pickle",
                              "wb") as pf:
                        pickle.dump(et_tmp_inher, pf, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f"gpt4-dump-inher\\100-everyday-things-{et}-hyponym-hypo_tmp_inher-gpt4.pickle",
                              "wb") as pf:
                        pickle.dump(hypo_tmp_inher, pf, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(f"gpt4-dump-inher\\100-everyday-things-{et}-hyponym-hypo_cond_et_tmp_inher-gpt4.pickle",
                              "wb") as pf:
                        pickle.dump(hypo_cond_et_tmp_inher, pf, protocol=pickle.HIGHEST_PROTOCOL)

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