import torch
from torch import nn


class CrossEntropyLabelSmooth(nn.Module):
    """
        This class implements label smoothing with cross-entropy loss as described in the paper 
        "Rethinking the Inception Architecture for Computer Vision". It can be used to improve 
        the generalization of the model by penalizing over-confident predictions.

        Label smoothing works by giving the model a target distribution that has a confidence of 
        (1 - epsilon) for the correct class and distributes epsilon across the other classes, 
        thus preventing the model from becoming too confident about a class which can help 
        in regularizing the model.

        Attributes:
            num_classes (int): The number of classes in the dataset.
            epsilon (float): The smoothing parameter epsilon, it specifies the amount of smoothing 
                            to apply (0 for no smoothing, 1 for complete smoothing).
            logsoftmax (nn.Module): A log softmax module configured to apply along the second 
                                    dimension (dim=1) of the input data.

        Args:
            num_classes (int): The number of classes in the dataset.
            epsilon (float): The smoothing parameter epsilon.
    """
    
	def __init__(self, num_classes, epsilon):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.logsoftmax = nn.LogSoftmax(dim=1)

	def forward(self, inputs, targets):
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		loss = (-targets * log_probs).mean(0).sum()
		return loss