from utils import compute

torch = compute.get_torch()
from data.TaskParams import TaskParams


class DataSetPostMapper:
    def __init__(self, task_params: TaskParams):
        self.dataset = task_params.dataset
        self.model = task_params.model
        self.tokenizer = task_params.tokenizer
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

            return {'probs': self._numpy(probs), 'predictions': self._numpy(predictions), 'correct': correct}
