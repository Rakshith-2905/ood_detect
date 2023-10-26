import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDINOLoss(nn.Module):
    def __init__(self, student_temp=0.1, teacher_temp=0.5):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp

    def forward(self, student_output, teacher_output):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # Teacher sharpening
        teacher_out = F.softmax(teacher_output / self.teacher_temp, dim=-1)
        teacher_out = teacher_out.detach()

        # CE(p, q) = -sum_{i} p_i * log(q_i)
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        return loss.mean()


# class SimpleDINOLoss(nn.Module):
#     def __init__(self, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs, nepochs, student_temp=0.1):
#         super().__init__()
#         self.student_temp = student_temp
#         # Temperature schedule for the teacher
#         self.teacher_temp_schedule = np.concatenate((
#             np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
#             np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
#         ))

#     def forward(self, student_output, teacher_output, epoch):
#         """
#         Cross-entropy between softmax outputs of the teacher and student networks.
#         """
#         student_out = student_output / self.student_temp

#         # Teacher sharpening based on epoch-specific temperature
#         temp = self.teacher_temp_schedule[epoch]
#         teacher_out = F.softmax(teacher_output / temp, dim=-1)
#         teacher_out = teacher_out.detach()

#         loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
#         return loss.mean()

def compute_accuracy(probs, labels):
    predictions = probs.argmax(dim=1)
    correct = (predictions == labels).float().sum()
    return (correct / probs.size(0)).item()


def compute_similarities(image_embeddings, text_embeddings, mode='cosine'):
    if mode == 'cosine':
        return cosine_similarities(image_embeddings, text_embeddings)
    elif mode == 'DN':
        return CLIP_DN_similarities(image_embeddings, text_embeddings)
    elif mode == 'DN*':
        cos_sim = cosine_similarities(image_embeddings, text_embeddings)
        dn_sim = CLIP_DN_similarities(image_embeddings, text_embeddings)
        return (cos_sim + dn_sim)/2

def cosine_similarities(image_embeddings, text_embeddings):
    """ Compute cosine similarities between image embeddings and text encodings for all labels """
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    similarities = image_embeddings @ text_embeddings.T
    return similarities

def CLIP_DN_similarities(image_embeddings, text_embeddings, image_embeddings_mean, text_embeddings_mean):
    "Compute cos similarity with distribution normalization"

    DN_image_embeddings = image_embeddings - image_embeddings_mean/2
    DN_text_embeddings = text_embeddings - text_embeddings_mean/2

    # Normalize the embeddings
    DN_image_embeddings = F.normalize(DN_image_embeddings, dim=-1)
    DN_text_embeddings = F.normalize(DN_text_embeddings, dim=-1)
    # Compute the dot product between the two tensors
    similarities = DN_image_embeddings.T @ DN_text_embeddings
    return similarities
 