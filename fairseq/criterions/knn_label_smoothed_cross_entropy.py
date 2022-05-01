import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import logging
logger = logging.getLogger(__name__)

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def knn_knowledge_distillation_nll_loss(lprobs, target, knn_targets, distances, ignore_index=None, reduce=True, knn_temp=1):
    knn_prob = F.softmax(-distances/knn_temp, dim=-1, dtype=lprobs.dtype)

    place_holder = torch.zeros_like(lprobs)
    knn_soft_target = place_holder.scatter_add_(1, knn_targets, knn_prob)# avoid  the error caused by in-place operate 
    KL_loss = F.kl_div(lprobs, knn_soft_target, reduction='none').sum(dim=-1).unsqueeze(-1)

    # print("knn_soft_target size is : ", knn_soft_target.size())
    # print("target index_select size is : ", target.size())
    # print("knn_targets size is : ", knn_targets.size())
    # print("knn_soft_target topk is : ", knn_soft_target[:100].topk(10))
    # knn_most_target = torch.mode(knn_targets, dim=-1).values
    # accurate_knn_target = torch.sum(torch.eq(knn_most_target, target.squeeze(-1)).type(torch.uint8))
    # print(f"knn mode accuracy : {accurate_knn_target.item()} / {target.size()}")
    # knn_prob_argmax = torch.argmax(knn_soft_target, dim=-1)
    # accurate_knn_prob_target = torch.sum(torch.eq(knn_prob_argmax, target.squeeze(-1)).type(torch.uint8))
    # print(f"knn prob accuracy : {accurate_knn_prob_target.item()} / {target.size()}")

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            KL_loss.masked_fill_(pad_mask, 0.)
    else:
        KL_loss = KL_loss.squeeze(-1)
    if reduce:
        KL_loss = KL_loss.sum()
    return KL_loss

def knn_adaptive_label_smoothed_nll_loss(ground_truth_loss, lprobs, target, knn_targets, distances, ignore_index=None, reduce=True, knn_temp=1, eta=0.0):

    knn_prob = F.softmax(-distances/knn_temp, dim=-1, dtype=lprobs.dtype)
    probs = torch.exp(lprobs).detach()
    place_holder = torch.zeros_like(lprobs)
    knn_soft_target = place_holder.scatter_add_(1, knn_targets, knn_prob)# avoid  the error caused by in-place operate 

    max_knn_prob = knn_soft_target.max(dim=-1, keepdim=True).values
    knn_max_factor = max_knn_prob / (1 + max_knn_prob) + eta
    p_max = probs.max(dim=-1, keepdim=True).values
    adaption_factor = torch.max(torch.cat((knn_max_factor, p_max), dim=-1), dim=-1, keepdim=True).values
    soft_target = (1 - adaption_factor) * knn_soft_target 
    soft_target.requires_grad = True
    knn_smoothed_loss = -torch.bmm(lprobs.unsqueeze(1), soft_target.unsqueeze(-1)).squeeze(-1)
    knn_adaptive_loss = adaption_factor * ground_truth_loss + knn_smoothed_loss
    
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            knn_adaptive_loss.masked_fill_(pad_mask, 0.)
    else:
        knn_adaptive_loss = knn_adaptive_loss.squeeze(-1)
    if reduce:
        knn_adaptive_loss = knn_adaptive_loss.sum()
    return knn_adaptive_loss

def knn_label_smoothed_nll_loss(lprobs, target, knn_targets, distances, ignore_index=None, reduce=True, knn_temp=1, eta=0.0):

    knn_prob = F.softmax(-distances/knn_temp, dim=-1, dtype=lprobs.dtype)
    place_holder = torch.zeros_like(lprobs)
    knn_soft_target = place_holder.scatter_add_(1, knn_targets, knn_prob)# avoid  the error caused by in-place operate 
    
    # vocab_size = lprobs.size(-1)
    # ground_truth_target = F.one_hot(target.squeeze(-1), vocab_size)
    # # knn_gt_prob = knn_soft_target.gather(dim=-1, index=target) # [batch_size * seqlen, 1]
    # # print("knn_gt_prob is : ", knn_gt_prob[:25])
    # max_knn_prob = knn_soft_target.max(dim=-1, keepdim=True).values
    # knn_max_factor = max_knn_prob / (1 + max_knn_prob) + eta
    # probs = torch.exp(lprobs).detach()
    # p_max = probs.max(dim=-1, keepdim=True).values
    # adaption_factor = torch.max(torch.cat((knn_max_factor, p_max), dim=-1), dim=-1, keepdim=True).values
    # soft_target = adaption_factor * ground_truth_target + (1 - adaption_factor) * knn_soft_target 
    # soft_target.requires_grad = True

    knn_soft_target.requires_grad = True
    knn_smoothed_loss = -torch.bmm(lprobs.unsqueeze(1), knn_soft_target.unsqueeze(-1)).squeeze(-1)

    # print("soft_target topk is : ", soft_target[:100].topk(10))
    # print("knn_soft_target size is : ", knn_soft_target.size())
    # print("soft_target size is : ", soft_target.size())
    # print("target index_select size is : ", target.size())
    # print("knn_targets size is : ", knn_targets.size())
    # knn_most_target = torch.mode(knn_targets, dim=-1).values
    # accurate_knn_target = torch.sum(torch.eq(knn_most_target, target.squeeze(-1)).type(torch.uint8))
    # print(f"knn mode accuracy : {accurate_knn_target.item()} / {target.size()}")
    # knn_prob_argmax = torch.argmax(knn_soft_target, dim=-1)
    # accurate_knn_prob_target = torch.sum(torch.eq(knn_prob_argmax, target.squeeze(-1)).type(torch.uint8))
    # print(f"knn prob accuracy : {accurate_knn_prob_target.item()} / {target.size()}")

    # soft_argmax = torch.argmax(soft_target, dim=-1)
    # accurate_soft_target = torch.sum(torch.eq(soft_argmax, target.squeeze(-1)).type(torch.uint8))
    # print(f"soft_target accuracy : {accurate_soft_target.item()} / {target.size()}")

    # print("nll_loss is : ", nll_loss)
    # print("knn_smoothed_loss is : ", knn_smoothed_loss)
    # print("mean nll loss is : ", torch.mean(nll_loss))
    # print("mean knn_smoothed_loss is : ", torch.mean(knn_smoothed_loss))

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        if pad_mask.any():
            knn_smoothed_loss.masked_fill_(pad_mask, 0.)
    else:
        knn_smoothed_loss = knn_smoothed_loss.squeeze(-1)
    if reduce:
        knn_smoothed_loss = knn_smoothed_loss.sum()
    return knn_smoothed_loss


@register_criterion('knn_label_smoothed_cross_entropy')
class KNNLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, knn_temp=1, temp_scheduling='fixed', linear_decay=20, inverse_sigmoid_decay=10, eta=0, distil_strategy='normal', kl_loss_combine_rate=1):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.knn_temp = knn_temp
        self.temp_scheduling = temp_scheduling
        self.linear_decay = linear_decay
        self.inverse_sigmoid_decay = inverse_sigmoid_decay
        self.eta = eta
        self.distil_strategy = distil_strategy
        self.kl_loss_combine_rate = kl_loss_combine_rate

    
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--knn-temp', type=float, default=1,
                            help='the knn distribution temperature')
        parser.add_argument('--temp-scheduling', type=str, default='fixed',
                            help='the knn temperature scheduling: gradually decreas the temperature')
        parser.add_argument('--linear-decay', type=float, default=20,
                            help='the knn temperature decay by linear scheduling')
        parser.add_argument('--inverse-sigmoid-decay', type=float, default=10,
                            help='the knn temperature decay by inverse sigmoid scheduling')
        parser.add_argument('--eta', type=float, default=0.1,
                            help='when adaptive smoothing weight, adaption_factor = max(p_max + lambda), lambda = (knn_max_probs) / (1 + knn_max_probs) + eta') 
        parser.add_argument('--distil-strategy', default='normal', help='set the strategy of the distillation')
        parser.add_argument('--kl-loss-combine-rate', default=1, type=float, help='set the kl loss combine rate')
        # fmt: on

    def forward(self, model, sample, epoch=1, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, epoch, reduce=reduce, distil_strategy=self.distil_strategy, temp_scheduling=self.temp_scheduling)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output
    
    def compute_loss(self, model, net_output, sample, epoch, reduce=True, distil_strategy="normal", temp_scheduling="fixed"):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample["target"].view(-1, 1)
        knntargets = sample["knntargets"]
        knntargets = knntargets.view(-1, knntargets.size(-1))
        distances = sample["distances"]
        distances = distances.view(-1, distances.size(-1))
        if temp_scheduling == "fixed":
            temp = self.knn_temp
        elif temp_scheduling == "linear":
            temp = max(1, self.knn_temp - (epoch - 1) * self.linear_decay)
        elif temp_scheduling == "inverse_sigmoid":
            temp = max(1, 2 * self.knn_temp * 1 / (1 + math.exp((epoch-1) / self.inverse_sigmoid_decay)))
        else:
            temp = self.knn_temp
        if distil_strategy == 'normal':
            # not use distillation
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
        elif distil_strategy == 'knn_only':
            _, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            loss = knn_label_smoothed_nll_loss(
                lprobs, target, knntargets, distances, ignore_index=self.padding_idx, reduce=reduce, knn_temp=temp, eta=self.eta
            )
        elif distil_strategy == 'knn_adaptive':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce='False',
            )
            nll_loss = nll_loss.sum()
            loss = knn_adaptive_label_smoothed_nll_loss(
                golden_loss, lprobs, target, knntargets, distances, ignore_index=self.padding_idx, reduce=reduce, knn_temp=temp, eta=self.eta
            )
        elif distil_strategy == 'knn_kd':
            golden_loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            kd_loss = knn_knowledge_distillation_nll_loss(
                lprobs, target, knntargets, distances, ignore_index=self.padding_idx, reduce=reduce, knn_temp=temp
            )
            loss = golden_loss + kd_loss * self.kl_loss_combine_rate
        else:
            logger.info(f"Unkown strategy {distil_strategy}, and use normal label smoothing.")
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
        return loss, nll_loss
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True