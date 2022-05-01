import logging
import os
import sys
from itertools import chain

import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
from tqdm import tqdm
import pickle
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")

def main(args, override_args=None):
    utils.import_user_module(args)

    assert (
            args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None
    # Load ensemble
    # the task is build based on the checkpoint
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(model_args)

    import numpy as np
    if args.knndistance_fp16:
        logger.info('Saving fp16')
        if args.store_in_normal_npy:
            dstore_distances = np.empty(shape=[args.knntarget_size, args.save_k], dtype=np.float16)
            dstore_targets = np.empty(shape=[args.knntarget_size, args.save_k], dtype=np.int32)
        else:
            dstore_distances = np.memmap(args.knntarget_mmap + f'/bs1_fp16_distances_{args.valid_subset}_k{args.save_k}.mmp', dtype=np.float16, mode='w+',
                                    shape=(args.knntarget_size, args.save_k))
            dstore_targets = np.memmap(args.knntarget_mmap + f'/bs1_knntargets_{args.valid_subset}_k{args.save_k}.mmp', dtype=np.int32, mode='w+',
                                    shape=(args.knntarget_size, args.save_k))
    else:
        logger.info('Saving fp32')
        if args.store_in_normal_npy:
            dstore_distances = np.empty(shape=[args.knntarget_size, args.save_k], dtype=np.float32)
            dstore_targets = np.empty(shape=[args.knntarget_size, args.save_k], dtype=np.int32)
        else:
            dstore_distances = np.memmap(args.knntarget_mmap + f'/bs1_fp32_distances_{args.valid_subset}_k{args.save_k}.mmp', dtype=np.float32, mode='w+',
                                    shape=(args.knntarget_size, args.save_k))
            dstore_targets = np.memmap(args.knntarget_mmap + f'/bs1_knntargets_{args.valid_subset}_k{args.save_k}.mmp', dtype=np.int32, mode='w+',
                                    shape=(args.knntarget_size, args.save_k))
    dstore_idx = 0
    # --- end
    data_idx = 1
    try:
        task.args.required_seq_len_multiple = 1
        task.args.load_alignments = False
        task.load_dataset(args.valid_subset, combine=False, epoch=data_idx)
        data_idx = data_idx + 1
        dataset = task.dataset(args.valid_subset)
    except KeyError:
        raise Exception("Cannot find dataset: " + args.valid_subset)
    
    # Initialize data iterator
    itr = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[m.max_positions() for m in models],
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        prefix=f"valid on '{args.valid_subset}' subset",
        default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
    )
    log_outputs = []
    knn_distances_pkl_list = []
    knn_targets_pkl_list = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            # -------- add by , we should go through the model with the sample and get the hidden state
            # so we append a forward_and_get_hidden_state_step method in Translation task
            features = task.forward_and_get_hidden_state_step(sample, model)  # [B, T, H]
        
            knn_search_result = model.decoder.knn_datastore.retrieve(features)
            knn_dists = knn_search_result['distance']  # [batch, seq len, save_k]  # we need do sort
            knn_index = knn_search_result['knn_index']
            tgt_index = knn_search_result['tgt_index']
            
            target = sample['target']  # [B, T]

            # get useful parameters
            batch_size = target.size(0)
            seq_len = target.size(1)
            pad_idx = task.target_dictionary.pad()
            # print("target is : ", target)
            target_mask = target.ne(pad_idx)  # [B, T]
            # print("target_mask is : ", target_mask)
            # remove the pad tokens and related hidden states
            target = target.view(batch_size * seq_len)
            target_mask = target_mask.view(batch_size * seq_len)
            # print("target size is : ", target.size())
            non_pad_index = target_mask.nonzero().squeeze(-1)  # [n_count]
            target = target.index_select(dim=0, index=non_pad_index)  # [n_count]

            features = features.contiguous().view(batch_size * seq_len, -1)
            features = features.index_select(dim=0, index=non_pad_index)  # [n_count, feature size]
            knn_dists = knn_dists.contiguous().view(batch_size * seq_len, -1)
            knn_dists = knn_dists.index_select(dim=0, index=non_pad_index)
            knn_index = knn_index.contiguous().view(batch_size * seq_len, -1)
            knn_index = knn_index.index_select(dim=0, index=non_pad_index)
            tgt_index = tgt_index.contiguous().view(batch_size * seq_len, -1)
            tgt_index = tgt_index.index_select(dim=0, index=non_pad_index)

            # print("target index_select size is : ", target.size())
            # print("tgt_index size is : ", tgt_index.size())
            # vocab_size = 42024
            # seq_len, k = tgt_index.size()
            # print("target is : ", target)
            # print("tgt_index is : ", tgt_index[:, 0])

            # knn_prob = F.softmax(-knn_dists/args.knn_temperature, dim=-1)
            # place_holder = torch.zeros(seq_len, vocab_size, dtype=knn_prob.dtype).to(tgt_index.device)
            # knn_soft_target = place_holder.scatter_add_(1, tgt_index, knn_prob)
            # knn_most_target = torch.mode(tgt_index, dim=-1).values
            # accurate_knn_target = torch.sum(torch.eq(knn_most_target, target.squeeze(-1)).type(torch.uint8))
            # print(f"mode accuracy : {accurate_knn_target.item()} / {target.size()}")
            # knn_prob_argmax = torch.argmax(knn_soft_target, dim=-1)
            # accurate_knn_prob_target = torch.sum(torch.eq(knn_prob_argmax, target.squeeze(-1)).type(torch.uint8))
            # print(f"prob accuracy : {accurate_knn_prob_target.item()} / {tgt_index.size()}")

            # if args.knndistance_fp16:
            #     knn_distances_pkl_list.append(list(knn_dists.detach().cpu().numpy().astype(np.float16)))
            #     knn_targets_pkl_list.append(list(tgt_index.cpu().numpy().astype(np.int32)))
            # else:
            #     knn_distances_pkl_list.append(list(knn_dists.detach().cpu().numpy().astype(
            #             np.float32)))
            #     knn_targets_pkl_list.append(list(tgt_index.cpu().numpy().astype(np.int32)))
          
            current_batch_count = target.size(0)
            if dstore_idx + current_batch_count > args.knntarget_size:
                reduce_size = args.knntarget_size - dstore_idx
                knn_dists = knn_dists[:reduce_size]
                knn_index = knn_index[:reduce_size]
                tgt_index = tgt_index[:reduce_size]
            else:
                reduce_size = current_batch_count

     
            if args.knndistance_fp16:
                dstore_distances[dstore_idx:reduce_size + dstore_idx] = knn_dists.detach().cpu().numpy().astype(
                    np.float16)
                dstore_targets[dstore_idx:reduce_size + dstore_idx] = tgt_index.cpu().numpy().astype(np.int32)
            else:
                dstore_distances[dstore_idx:reduce_size + dstore_idx] = knn_dists.detach().cpu().numpy().astype(
                    np.float32)
                dstore_targets[dstore_idx:reduce_size + dstore_idx] = tgt_index.cpu().numpy().astype(np.int32)
            dstore_idx += reduce_size

            # print(dstore_idx)
            if dstore_idx > args.knntarget_size:
                logger.info('much more than knntarget size break')
                break
            if i % 10000 == 0:
                logger.info(f"Finished {i} / {len(progress)}")
    if args.store_in_normal_npy:
        if args.knndistance_fp16:
            np.save(args.knntarget_mmap + f'/bs1_fp16_distances_{args.valid_subset}_k{args.save_k}.npy', dstore_distances)
        else:
            np.save(args.knntarget_mmap + f'/bs1_fp32_distances_{args.valid_subset}_k{args.save_k}.npy', dstore_distances)
        np.save(args.knntarget_mmap + f'/bs1_knntargets_{args.valid_subset}_k{args.save_k}.npy', dstore_targets)

    # if args.knndistance_fp16:
    #     with open(args.knntarget_mmap + f'/fp16_distances_{args.valid_subset}.pkl', 'wb') as writer:
    #         pickle.dump(knn_distances_pkl_list, writer)
    #     with open(args.knntarget_mmap + f'/knntargets_{args.valid_subset}.pkl', 'wb') as writer:
    #         pickle.dump(knn_targets_pkl_list, writer)
    # else:
    #     with open(args.knntarget_mmap + f'/fp32_distances_{args.valid_subset}.pkl', 'wb') as writer:
    #         pickle.dump(knn_distances_pkl_list, writer)
    #     with open(args.knntarget_mmap + f'/knntargets_{args.valid_subset}.pkl', 'wb') as writer:
    #         pickle.dump(knn_targets_pkl_list, writer)


def cli_main():
    parser = options.get_save_knntargets_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_save_knntargets_parser()
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)



if __name__ == "__main__":
    cli_main()
