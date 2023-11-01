import os
# import openai
import numpy as np
import pandas as pd
import argparse
from typing import Optional
from transformers import pipeline, set_seed

def load_opt_model():
    set_seed(32)
    generator = pipeline('text-generation', model="facebook/opt-1.3b", do_sample=True)
    return generator

def get_answers(generator, prompt, context: Optional[str] = None):
    if context: prompt = f"{prompt}. You are given the context {context}\n"
    ans = generator(prompt)
    return ans





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, required=True, default='bamboogle')
    parser.add_argument('--data_dir', type=str, required=True, default='../Data')
    parser.add_argument('--num_test', type=int, required=True, default=-1)
    parser.add_argument('--use_open_ai', type=bool, required=False, default=False)
    parser.add_argument('--use_opt_model', type=bool, required=False, default=True)



    args = parser.parse_args()
    if args.data_name == 'bamboogle':
        if args.data_dir:
            file_names = os.listdir(args.data_dir)
            data = pd.DataFrame()
            for file in file_names:
                file_path = os.path.join(args.data_dir, file)
                current_data = pd.read_csv(file_path)
                data = pd.concat([data, current_data], ignore_index=True)
        print(f'Loaded {args.data_name} data')
        if args.num_test!=-1:
            rand_indices = np.random.choice(len(data), args.num_test, replace=False)
            test_data = data.iloc[list(rand_indices)]
            train_data = data.drop(list(rand_indices))
            print(len(test_data), len(train_data))
        else:
            test_data = data
        if args.use_opt_model:
            generator = load_opt_model()
            train_data['PredictedAnswer'] = train_data['Questions'].apply(lambda row: get_answers(generator, row))
            print(f"Finished processing Train {args.data_name}")
            test_data['PredictedAnswer'] = test_data['Questions'].apply(lambda row: get_answers(generator, row))
            print(f"Finished processing Test {args.data_name}")
            print(f"Saving Outputs...")
            test_data.to_csv(os.path.join(args.data_dir,f'predicted_test_data_{args.data_name}.csv'))
            train_data.to_csv(os.path.join(args.data_dir, f"predicted_train_data_{args.data_name}.csv"))






if __name__ == '__main__':
    main()


