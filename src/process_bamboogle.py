import os
from ast import literal_eval

# import openai
import torch
import numpy as np
import pandas as pd
import argparse
from typing import Optional
from tqdm import tqdm
from transformers import pipeline, set_seed

tqdm.pandas()


def load_openLlama_model(load_optimized: bool = True):
    set_seed(32)
    if load_optimized:
        generator = pipeline(
            "text-generation",
            model="openlm-research/open_llama_3b_v2",
            do_sample=True,
            return_full_text=False,
            torch_dtype=torch.bfloat16,
            max_length=40,
        )
    else:
        generator = pipeline(
            "text-generation",
            model="openlm-research/open_llama_3b_v2",
            do_sample=True,
            return_full_text=False,
            max_length=40,
        )
    return generator


def load_opt_model(load_optimized: bool = True):
    set_seed(32)
    if load_optimized:
        generator = pipeline(
            "text-generation",
            model="facebook/opt-1.3b",
            do_sample=True,
            return_full_text=False,
            torch_dtype=torch.bfloat16,
            max_length=40,
        )
    else:
        generator = pipeline(
            "text-generation",
            model="facebook/opt-1.3b",
            do_sample=True,
            return_full_text=False,
            max_length=40,
        )
    return generator


def load_gpt2_model(load_optimized: bool = True):
    set_seed(32)
    if load_optimized:
        generator = pipeline(
            "text-generation",
            model="gpt2",
            do_sample=True,
            return_full_text=False,
            torch_dtype=torch.bfloat16,
            max_length=40,
        )
    else:
        generator = pipeline(
            "text-generation",
            model="gpt2",
            do_sample=True,
            return_full_text=False,
            max_length=40,
        )
    return generator


def load_few_shot_prompt(few_shot_file_path: str):
    with open(few_shot_file_path, "r") as f:
        few_shot_prompt = f.read()
    return few_shot_prompt


def get_answers(
    generator,
    prompt,
    use_few_shot_prompt: Optional[bool] = None,
    context: Optional[str] = None,
):
    if context:
        prompt = f"{prompt}. You are given the context {context}\n"
        if use_few_shot_prompt:
            few_shot_prompt = load_few_shot_prompt(
                "RAG-Research/data/cot_bamboogle.txt"
            )
            prompt += f"You are given the following pairs of questions and answers as references: \n {few_shot_prompt}"
    ans = generator(prompt)
    return ans


def tokenize(a):
    """
    lower, split, strip each token
    """
    b = a.lower().split()
    for ii in range(len(b)):
        b[ii] = b[ii].strip().strip("?.,\"'").strip()
    return b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True, default="bamboogle")
    parser.add_argument("--data_dir", type=str, required=True, default="./Data")
    parser.add_argument("--num_test", type=int, required=True, default=-1)
    parser.add_argument("--use_open_ai", type=bool, required=False, default=False)
    parser.add_argument(
        "--get_predictions", dest="get_predictions", action="store_true"
    )
    parser.add_argument(
        "--get_eval_results", dest="get_eval_results", action="store_true"
    )
    parser.add_argument("--use_opt_model", dest="get_eval_results", action="store_true")
    parser.add_argument(
        "--use_few_shot_prompt", dest="use_few_shot_prompt", action="store_true"
    )
    parser.add_argument(
        "--model_name", dest="model_name", choices=["opt", "gpt", "openLlama"]
    )

    args = parser.parse_args()
    if args.data_name == "bamboogle" and args.get_predictions:
        if args.data_dir:
            file_names = os.listdir(args.data_dir)
            data = pd.DataFrame()
            for file in file_names:
                file_path = os.path.join(args.data_dir, file)
                current_data = pd.read_csv(file_path)
                data = pd.concat([data, current_data], ignore_index=True)
        print(f"Loaded {args.data_name} data")

        if args.num_test != -1:
            rand_indices = np.random.choice(len(data), args.num_test, replace=False)
            test_data = data.iloc[list(rand_indices)]
            train_data = data.drop(list(rand_indices))
            print(
                f"Number of samples in Test Data: {len(test_data)}, Number of samples in Train data{len(train_data)}"
            )
        else:
            test_data = data
        load_optimized = False
        if args.model_name == "opt":
            generator = load_opt_model(load_optimized)
        if args.model_name == "gpt":
            generator = load_gpt2_model(load_optimized)
        if args.model_name == "openLlama":
            generator = load_openLlama_model(load_optimized)
        train_data["PredictedAnswer"] = train_data["Question"].progress_apply(
            lambda row: get_answers(generator, row)
        )
        print(f"Finished processing Train {args.data_name}")
        test_data["PredictedAnswer"] = test_data["Question"].progress_apply(
            lambda row: get_answers(generator, row)
        )
        print(f"Finished processing Test {args.data_name}")
        print(f"Saving Outputs...")
        test_data.to_csv(
            os.path.join(args.data_dir, f"predicted_test_data_{args.data_name}.csv")
        )
        train_data.to_csv(
            os.path.join(args.data_dir, f"predicted_train_data_{args.data_name}.csv")
        )

    if args.get_eval_results:
        F1_list = []
        InterRecall_list = []

        if os.path.exists(
            os.path.join(args.data_dir, f"predicted_test_data_{args.data_name}.csv")
        ):
            data_test = pd.read_csv(
                os.path.join(args.data_dir, f"predicted_test_data_{args.data_name}.csv")
            )
            data_train = pd.read_csv(
                os.path.join(
                    args.data_dir, f"predicted_train_data_{args.data_name}.csv"
                )
            )
            data = pd.concat([data_test, data_train], ignore_index=True).reset_index()
            data["PredictedAnswer"] = data["PredictedAnswer"].apply(
                lambda x: literal_eval(x)
            )
            for i, current_data in data.iterrows():
                output_w = set(
                    tokenize(current_data["PredictedAnswer"][0]["generated_text"])
                )
                target_w = set(tokenize(current_data["Answer"]))
                num_share_w = len(output_w & target_w)
                if num_share_w == 0:
                    f1 = 0
                else:
                    precision = num_share_w / len(output_w)
                    recall = num_share_w / len(target_w)
                    f1 = 2 * precision * recall / (precision + recall)
                F1_list.append(f1)

        assert len(F1_list) != 0, "F1 list is empty"
        print(f"F1 score :::: {np.mean(F1_list)}")


if __name__ == "__main__":
    main()
