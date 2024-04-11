import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from datasets import load_dataset
import torch.optim as optim

# code
# https://pytorch.org/docs/stable/_modules/torch/optim/lbfgs.html#LBFGS.step
# https://www.programcreek.com/python/example/92671/torch.optim.LBFGS


print("Transformers Version: " + f"{transformers.__version__}")
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')

# replacing with any dataset I need
# datasets = load_dataset("text", data_files={"train": path_to_train.txt, "validation": path_to_validation.txt}

model_checkpoint = "distilgpt2"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

tokenized_datasets["train"][1] # We have the input_ids now 

# block_size = tokenizer.model_max_length
block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# Model
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    f"{model_name}-finetuned-wikitext2",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
)

class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        # Create your custom optimizer
        self.optimizer = optim.LBFGS(self.model.parameters())

        # You can also customize the learning rate scheduler
        self.lr_scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )

    def training_step(self, model, inputs):
        model.train()
        for key, value in inputs.items():
            inputs[key] = value.to(self.args.device)

        def closure():
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            loss.backward()
            return loss

        self.optimizer.step(closure)

        return closure()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"]
)

trainer.train()

import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")