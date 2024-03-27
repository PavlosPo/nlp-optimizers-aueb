import torch
from tqdm import tqdm

# Make a class evaluation that will behave like the Evaluate API from Hugging Face
class CustomEvaluator:
    def __init__(self, original_model, functional_model, params, buffers, test_loader, metric):
        self.original_model = original_model
        self.functional_model = functional_model
        self.params = params
        self.buffers = buffers
        self.test_loader = test_loader
        self.evaluation_results = []
        self.metric = metric
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate(self):
        self.original_model.eval()
        # Set model to evaluation mode
        # self.functional_model.eval()
        # self.functional_model.to(self.device)
        
        # Evaluate model
        self.original_model.eval()
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.test_loader, 1), total=len(self.test_loader))
            for i, data in progress_bar:
                evaluation_result = {}
                progress_bar.set_description(f'Step {i}/{len(self.test_loader)}')
                
                input_ids = data['input_ids'].squeeze().to(self.device)
                attention_mask = data['attention_mask'].squeeze().to(self.device)
                labels = data['labels'].squeeze().to(self.device)
                
                # Get model predictions
                preds = self.functional_model(self.params, self.buffers, input_ids, attention_mask)

                predictions = torch.round(preds) # Round predictions to 0 or 1 - Sigmoid Activation function
                
                # Save the evaluation results
                evaluation_result['input_ids'] = input_ids.cpu().tolist()
                evaluation_result['attention_mask'] = attention_mask.cpu().tolist()
                evaluation_result['labels'] = labels.cpu().tolist()
                evaluation_result['preds'] = preds.cpu().tolist()
                evaluation_result['predictions'] = predictions.cpu().tolist()

                self.evaluation_results.append(evaluation_result)
                self.metric.add_batch(predictions=predictions, references=labels)
        
        results = self.metric.compute()
        print(f"Results: \n{results}\n")

        print('Finished Training')

        print(f"\nLabels: {self.evaluation_results[0]['labels']}\n")
        print(f"\nPreds: {self.evaluation_results[0]['preds']}")
        print(f"\nPredicted Labels: {self.evaluation_results[0]['predictions']}")
        
        return self.evaluation_results
