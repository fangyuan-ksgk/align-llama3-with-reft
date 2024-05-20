from huggingface_hub import login
login(
  token=HF_TOKEN, # ADD YOUR TOKEN HERE
  add_to_git_credential=True
)

import torch
from tqdm import tqdm as tqdm
import numpy as np
import transformers
import pyreft 
from pyreft import ReftModel
from datasets import load_dataset
import datasets
device = "cuda" if torch.cuda.is_available() else "cpu"


def prepare_sale_dataset():
    from data.corpus import mock_conversation

    dataset_dict = {
        "system_prompt": [],
        "prompt": [],
        "completion": []
    }

    for i_round in range(len(mock_conversation)//2):
        mock_history = mock_conversation[:2*i_round + 1]
        customer_utterance = mock_conversation[2*i_round + 1]
        sale_response = mock_conversation[2*i_round + 2]

        system_prompt = "[Conversation History] " + " ".join(mock_history) + "[End]" 
        prompt = "[Customer Utterance] " + customer_utterance + " [Sale Response] "
        completion = sale_response

        dataset_dict["system_prompt"].append(system_prompt)
        dataset_dict["prompt"].append(prompt)
        dataset_dict["completion"].append(completion)

    # prepare a huggingface dataset object
    dataset = datasets.Dataset.from_dict(dataset_dict)

    return dataset



########################
# Load Llama3-8B model #
########################

model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

model_max_length = 4096
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=model_max_length, 
    padding_side="right", use_fast=False)
if "Meta-Llama-3-" in model_name_or_path:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
else:
    tokenizer.pad_token = tokenizer.unk_token

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

####################
# Setup Reft model #
####################

reft_config = pyreft.ReftConfig(representations=[{
    "layer": l, "component": "block_output",
    "low_rank_dimension": 2,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=2)} for l in [8, 16, 24]])
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device("cuda")
reft_model.print_trainable_parameters()

# position info about the interventions
share_weights = True # whether the prefix and suffix interventions sharing weights.
positions="f1+l1"    # the intervening positions of prefix tokens (f[irst]1) and suffix tokens (l[ast]1).
first_n, last_n = pyreft.parse_positions(positions)


###############
# Data Module # 
###############

dataset = prepare_sale_dataset()

query_list = [tokenizer.apply_chat_template(
        [
            {"role": "system", "content": data["system_prompt"]}, 
            {"role": "user", "content": data['prompt']}
        ], tokenize=False
) for data in dataset]

answer_list = [
        tokenizer.apply_chat_template(
            [{"role": "assistant", "content": data['completion']}], tokenize=False,
        )[len(tokenizer.bos_token):] for data in dataset]

data_module = pyreft.make_multiple_position_supervised_data_module(
    tokenizer, model, query_list, answer_list, 
    positions=positions, num_interventions=len(reft_config.representations), share_weights=share_weights, nonstop=False)

################
# Train & Save #
################

training_args = transformers.TrainingArguments(
    num_train_epochs=50.0, output_dir="./tmp", 
    per_device_train_batch_size=10, 
    learning_rate=4e-3, report_to=[], logging_steps=20)
trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model, tokenizer=tokenizer,
    args=training_args, **data_module)
_ = trainer.train()

reft_model.set_device("cpu") # send back to cpu before saving.
reft_model.save(
    save_directory="./reft_to_share", 
    save_to_hf_hub=True, 
    hf_repo_name="Ksgk-fy/agent_clone_reft_llama3_01"
)

##############
# Evaluation #
##############

# def run_eval(dataset, tokenizer, reft_model):
#     pd = tqdm(total=len(dataset), position=0, leave=False)
#     n_correct = 0
#     for data in dataset:
#         # tokenize and prepare the input
#         prompt = tokenizer.apply_chat_template(
#             [{"role": "system", "content": system_prompt}, {"role": "user", "content": data['prompt']}], 
#             tokenize=False)
#         prompt = tokenizer(prompt, return_tensors="pt").to(device)
        
#         unit_locations = torch.IntTensor([pyreft.get_intervention_locations(
#             last_position=prompt["input_ids"].shape[-1], 
#             first_n=first_n, 
#             last_n=last_n,
#             pad_mode="last",
#             num_interventions=len(reft_config.representations),
#             share_weights=share_weights
#         )]).permute(1, 0, 2).tolist()
        
#         _, reft_response = reft_model.generate(
#             prompt, unit_locations={"sources->base": (None, unit_locations)},
#             intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
#             eos_token_id=terminators, early_stopping=True
#         )
#         response = tokenizer.decode(reft_response[0])
#         pred = response.split('<|start_header_id|>assistant<|end_header_id|>\n\n')[1].split('<|eot_id|>')[0]
#         gt = data['completion']
#         print("Prediction: ", pred, " GT: ", gt)
#         if pred == gt:
#             n_correct += 1
#         pd.update(1)
#     correct_rate = n_correct / len(dataset)
#     print("Correct Rate: ", np.round(correct_rate * 100, 1), "%")
#     return correct_rate


# run_eval(dataset, tokenizer, reft_model)