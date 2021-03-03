from utils import compute

torch = compute.get_torch()


class DataSetPostMapper:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = compute.get_device()

    def _tensor(self, lst):
        return torch.tensor(lst, device=self.device)

    def _numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def add_is_correct_and_probs(self, examples):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._tensor(examples['input_ids']),
                                self._tensor(examples['attention_mask'])).logits
            probs = logits.softmax(dim=1).max(dim=1).values
            predictions = logits.argmax(dim=1)
            correct = [1 if y_hat == y else 0 for y, y_hat in
                       zip(examples['label'], predictions)]

            return {'prob': self._numpy(probs), 'prediction': self._numpy(predictions), 'correct': correct}

    def add_probs(self, examples):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._tensor(examples['input_ids']),
                                self._tensor(examples['attention_mask'])).logits
            probs = logits.softmax(dim=1).max(dim=1).values

            return {'prob': self._numpy(probs)}

    def change_labels(self, examples, field_name='correct'):
        return {'label': examples[field_name]}
