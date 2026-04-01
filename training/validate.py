from typing import Union
import os
import torch
import torch.distributed as dist
from config import Config
from utils import *
from .postprocess import process_outputs,ResultSaver
import json


def validate(
    args, tasks,model, loss_fn, val_loader, epoch, device, testing=False
) -> Union[float, dict]:
    
    model.eval()
    
    model_labels,tgts_trans_for_loss,outs_trans_for_loss, outs_trans_for_res = Config.get_model_config_(
            args.model_name,"labels","targets_transform_for_loss","outputs_transform_for_loss", "outputs_transform_for_results"
        )

    val_loss_sum = 0.0
    total_samples = 0
    
    local_results = {task: [] for task in tasks}
    local_targets = {task: [] for task in tasks}
    
    if testing and args.save_test_results and is_main_process():
        results_saver = ResultSaver(item_names=tasks)
    else:
        results_saver = None
    
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 500
    # import numpy as np
    # timings=np.zeros((repetitions,1))

    sampling_rate = val_loader.dataset.sampling_rate()

    with torch.no_grad():
        progress = None
        if is_main_process():
            if testing:
                prefix = "Test: "
            else:
                prefix = f"Val: [{epoch}/{args.epochs}]"
            progress = ProgressMeter(
                len(val_loader),
                [],
                prefix=prefix,
            )

        for step, (x, loss_targets, metrics_targets, meta_data_jsons) in enumerate(val_loader):
            if x.size(0) == 0:
                if is_dist_avail_and_initialized():
                    dist.barrier() 
                continue

            current_device = model.device

            if isinstance(x, (list, tuple)):
                x = [xi.to(current_device).bfloat16() for xi in x]
            else:
                x = x.to(current_device).bfloat16()

            if isinstance(loss_targets, (list, tuple)):
                loss_targets = [yi.to(current_device) for yi in loss_targets]
            else:
                loss_targets = loss_targets.to(current_device)

            # if step >= 100  and step < 600:
            #     starter.record()

            # metadata = None
            # if isinstance(meta_data_jsons, tuple) and len(meta_data_jsons) > 0:
            #     metadata = meta_data_jsons
            
            # outputs = model(x, metadata=metadata)
            outputs = model(x, metadata=meta_data_jsons)
            
            # if step >= 100 and step < 600:
            #     ender.record()
            #     torch.cuda.synchronize()
            #     curr_time = starter.elapsed_time(ender)
            #     timings[step-100] = curr_time

            # Loss
            outputs_for_loss = outs_trans_for_loss(outputs) if outs_trans_for_loss is not None else outputs
            loss_targets = tgts_trans_for_loss(loss_targets) if tgts_trans_for_loss is not None else loss_targets

            loss = loss_fn(outputs_for_loss, loss_targets)

            # Batch size of this step
            step_batch_size = x.size(0)
            val_loss_sum += loss.item() * step_batch_size
            total_samples += step_batch_size

            # Process outputs
            outputs_for_metrics = outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
            results = process_outputs(args, outputs_for_metrics,model_labels,sampling_rate)

            if results_saver is not None:
                if isinstance(meta_data_jsons,torch.Tensor):
                    meta_data_jsons = meta_data_jsons.detach().cpu().tolist()
                
                meta_data_dict={k:[] for k in json.loads(meta_data_jsons[0]).keys()}
                for j in meta_data_jsons:
                    for k,v in json.loads(j).items():
                        meta_data_dict[k].append(v)
                results_saver.append(meta_data_dict,metrics_targets,results)
            
            for task in tasks:
                local_results[task].append(results[task].cpu())
                local_targets[task].append(metrics_targets[task].cpu())
            
            if is_main_process() and step % args.log_step == 0:
                prg_str = progress.get_str(batch_idx=step, name=f"{args.model_name}_{'test' if testing else 'val'}")
                logger.info(prg_str)

    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # print(f'Mean step time: {mean_syn:.2f} ms, std: {std_syn}')

    if is_dist_avail_and_initialized():
        dist.barrier()
        
    # Sum the loss and total_samples across all GPUs
    total_loss_tensor = torch.tensor([val_loss_sum, total_samples], dtype=torch.float64, device=device)
    if is_dist_avail_and_initialized():
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    
    final_avg_loss = (total_loss_tensor[0] / total_loss_tensor[1]).item()

    world_size = get_world_size()
    
    final_metrics_dict = {}
    for task in tasks:
        gathered_results = [None] * world_size
        gathered_targets = [None] * world_size

        if is_dist_avail_and_initialized():
            dist.all_gather_object(gathered_results, local_results[task])
            dist.all_gather_object(gathered_targets, local_targets[task])
        else:
            gathered_results = [local_results[task]]
            gathered_targets = [local_targets[task]]

        if is_main_process():
            all_preds = torch.cat([pred for rank_preds in gathered_results for pred in rank_preds])
            all_targets = torch.cat([target for rank_targets in gathered_targets for target in rank_targets])

            metrics = Metrics(
                task=task,
                metric_names=Config.get_metrics(task),
                sampling_rate=sampling_rate,
                time_threshold=args.time_threshold,
                num_samples=args.in_samples,
                device=device, 
            )
            metrics.compute(
                targets=all_targets.to(device),
                preds=all_preds.to(device),
                reduce=False, 
            )
            final_metrics_dict[task] = metrics

    if results_saver is not None:
        results_save_path = get_safe_path(os.path.join(logger.logdir(),f"test_results_{val_loader.dataset.name()}.csv"))
        results_saver.save_as_csv(results_save_path)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return final_avg_loss, final_metrics_dict
