from model import BertClassifier
from dataset import CustomDataLoader
from training import CustomTrainer
from evaluation import CustomEvaluator
from utils import set_seed

# Constants
SEED_NUM = 42
EPOCHS = 2

# Set seed
set_seed(SEED_NUM)

# Load model
original_model = BertClassifier(
              model_name='distilbert-base-uncased', 
              num_classes=1)

# Prepare dataset
custom_dataloader = CustomDataLoader(
    dataset_from='glue',
    model_name='bert-base-uncased',
    dataset_task='sst2',
    seed_num=SEED_NUM,
    range_to_select=100,
    batch_size=8
)
train_loader, test_loader, metric = custom_dataloader.get_custom_data_loaders()

# Train model
trainer = CustomTrainer(original_model, train_loader, epochs=EPOCHS)

functional_model, params, buffers = trainer.train() # Get functional model, params and buffers

# Evaluate model
evaluator = CustomEvaluator(
    original_model=original_model,
    functional_model=functional_model,
    params=params,
    buffers=buffers,
    test_loader=test_loader,
    metric=metric
)

evaluation_results = evaluator.evaluate()

print("Evaluation results:", evaluation_results)
