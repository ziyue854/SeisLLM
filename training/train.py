import datetime
import inspect
import math
import os
import shutil
from typing import Union
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from config import Config
from models import create_model, load_checkpoint, save_checkpoint
from utils import *
from .postprocess import process_outputs
from .preprocess import SeismicDataset
from .validate import validate
import torch._dynamo
import flash_attn
import deepspeed
torch._dynamo.config.suppress_errors = True


def train(
    args,
    tasks,
    model,
    optimizer,
    scheduler,
    loss_fn,
    train_loader,
    epoch,
    device,
    tensor_writer,
) -> Union[float, dict]:
    model.train()
    device = model.device

    # Save and display metrics
    train_loss_per_step = []
    average_meters = {}
    metrics_merged = {}
    sampling_rate = train_loader.dataset.sampling_rate()

    for task in tasks:
        metrics = Metrics(
            task=task,
            metric_names=Config.get_metrics(task),
            sampling_rate=sampling_rate,
            time_threshold=args.time_threshold,
            num_samples=args.in_samples,
            device=device,
        )
        metrics_merged[f"{task}"] = metrics
        for metric in metrics.metric_names():
            average_meters[f"{task}_{metric}"] = AverageMeter(
                f"[{task.upper()}]{metric}", ":6.4f"
            )

    average_meters["loss"] = AverageMeter("Loss", ":6.4f")
    progress = ProgressMeter(
        len(train_loader),
        [m for m in average_meters.values()],
        prefix=f"Train: [{epoch}/{args.epochs}]",
    )#math.ceil(len(train_loader) / model.gradient_accumulation_steps()),

    (
        label_names,
        tgts_trans_for_loss,
        outs_trans_for_loss,
        outs_trans_for_res,
    ) = Config.get_model_config_(
        args.model_name,
        "labels",
        "targets_transform_for_loss",
        "outputs_transform_for_loss",
        "outputs_transform_for_results",
    )

    for step, batch_data in enumerate(train_loader):
        if isinstance(batch_data, list):
            x, loss_targets, metrics_targets, metadata_batch = batch_data
        else:
            x, loss_targets, metrics_targets, metadata_batch = batch_data
        
        if isinstance(x, (list, tuple)):
            x = [xi.to(device).bfloat16() for xi in x]
        else:
            x = x.to(device).bfloat16()

        if isinstance(loss_targets, (list, tuple)):
            loss_targets = [yi.to(device) for yi in loss_targets]
        else:
            loss_targets = loss_targets.to(device)

        # Forward
        outputs = model(x, metadata=metadata_batch)
           
        # Loss
        outputs_for_loss = (
            outs_trans_for_loss(outputs) if outs_trans_for_loss is not None else outputs
        )
        loss_targets = (
            tgts_trans_for_loss(loss_targets)
            if tgts_trans_for_loss is not None
            else loss_targets
        )
        loss = loss_fn(outputs_for_loss, loss_targets)

        # Backward and Adjust learning rate for every step instead of epoch 
        model.backward(loss)
        model.step()
        lr = model.get_lr()[0]

        # Batch size of the step
        step_batch_size = x.size(0)
        # Reduce
        if is_dist_avail_and_initialized():
            loss = reduce_tensor(loss, "AVG")
            step_batch_size = torch.tensor(
                step_batch_size, device=device, dtype=torch.int32
            )
            step_batch_size = reduce_tensor(step_batch_size)
            dist.barrier()
            step_batch_size = step_batch_size.item()

        # Save loss
        average_meters["loss"].update(loss.item(), step_batch_size)
        train_loss_per_step.append(loss.item())

        # Process outputs
        outputs_for_metrics = (
            outs_trans_for_res(outputs) if outs_trans_for_res is not None else outputs
        )
        results = process_outputs(args, outputs_for_metrics, label_names, sampling_rate)

        # Calculate metrics
        tasks_metrics = {}
        for task in tasks:
            metrics = Metrics(
                task=task,
                metric_names=Config.get_metrics(task),
                sampling_rate=sampling_rate,
                time_threshold=args.time_threshold,
                num_samples=args.in_samples,
                device=device,
            )
            tasks_metrics[task] = metrics
            metrics.compute(
                targets=metrics_targets[task],
                preds=results[task],
                reduce=is_dist_avail_and_initialized(),
            )
            for metric in metrics.metric_names():
                average_meters[f"{task}_{metric}"].update(
                    metrics.get_metric(name=metric), step_batch_size
                )
            metrics_merged[f"{task}"].add(metrics)

        # Tensorboard
        if tensor_writer is not None and is_main_process():
            gstep = epoch * len(train_loader) + step
            # tensor_writer.add_scalar("learning-rate/step", lr, gstep)
            tensor_writer.add_scalar("train-loss/step", loss.item(), gstep)
            for task in tasks:
                values = tasks_metrics[task].get_all_metrics()
                tensor_writer.add_scalars(f"train.{task}.metrics/step", values, gstep)

        if step % args.log_step == 0 and is_main_process():
            prg_str = progress.get_str(batch_idx=step, name=f"{args.model_name}_train")
            logger.info(prg_str)

    return train_loss_per_step, metrics_merged


def train_worker(args, device) -> str:
    # Log
    logger.set_logger("train")

    log_dir = logger.logdir()
    checkpoint_save_dir = get_safe_path(os.path.join(log_dir, "checkpoints"))
    tb_dir = get_safe_path(os.path.join(log_dir, "tensorboard"))

    tensor_writer = SummaryWriter(tb_dir) if args.use_tensorboard else None

    if is_main_process():
        with open(os.path.join(log_dir, f"run_tb_{get_time_str()}.sh"), "w") as f:
            f.write(f"tensorboard --logdir '{tb_dir}' --port 8080")
        if not os.path.exists(checkpoint_save_dir):
            os.makedirs(checkpoint_save_dir)

    # Data loader
    model_inputs, model_labels, model_tasks = Config.get_model_config_(
        args.model_name, "inputs", "labels", "eval"
    )
    in_channels = Config.get_num_inchannels(model_name=args.model_name)

    train_dataset = SeismicDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="train",
    )
    val_dataset = SeismicDataset(
        args=args,
        input_names=model_inputs,
        label_names=model_labels,
        task_names=model_tasks,
        mode="val",
    )

    logger.info(f"train size: {len(train_dataset)}, val size:{len(val_dataset)}")

    train_sampler = (
        torch.utils.data.DistributedSampler(train_dataset)
        if is_dist_avail_and_initialized()
        else None
    )
    val_sampler = (
        torch.utils.data.DistributedSampler(val_dataset)
        if is_dist_avail_and_initialized()
        else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=((not is_dist_avail_and_initialized()) and args.shuffle),
        pin_memory=args.pin_memory,
        num_workers=args.workers,
        sampler=val_sampler,
    )

    # Epochs & Steps
    if args.steps > 0:
        args.epochs = math.ceil(args.steps / len(train_loader))
    args.steps = args.epochs * len(train_loader)
    logger.warning(f"`args.epochs` -> {args.epochs}, `args.steps` -> {args.steps}")

    logger.info("Running in DeepSpeed mode...")

    true_sampling_rate = 50 
    if hasattr(train_dataset, 'sampling_rate') and callable(train_dataset.sampling_rate):
        true_sampling_rate = train_dataset.sampling_rate()

    # Model
    model = create_model(
        model_name=args.model_name,
        in_channels=in_channels,
        in_samples=args.in_samples,
        llm_type=args.llm,
        llm_layers=args.LLM_layer_num,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dataset_name = args.dataset_name,
        n_heads=args.n_heads,
        sampling_rate=true_sampling_rate,
        use_flash_attn_2=args.use_flash_attn,
    )

    # model.llm_block = model.llm_block.to(torch.float16)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        args=args, model=model, model_parameters=params_to_optimize, training_data=train_dataset
    )
    device = model_engine.device
    model = model_engine

    if args.checkpoint:
        _, client_state = model.load_checkpoint(args.checkpoint)
        args.start_epoch = client_state.get('epoch', 0) + 1
        best_loss = client_state.get('best_loss', float('inf'))
    else:
        best_loss = float('inf')
    loss_fn = Config.get_loss(model_name=args.model_name).to(device)

    # Save loss
    losses_dict = {
        n: []
        for n in ["train_loss_per_step", "train_loss_per_epoch", "val_loss_per_epoch"]
    }

    num_saved = 0
    epochs_since_improvement = 0
    ckpt_path = None
    cost_time = datetime.timedelta()

    for i, epoch in enumerate(range(args.start_epoch, args.epochs)):
        epoch_start_time = datetime.datetime.now()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch=epoch)

        # Train
        train_losses, train_metrics_dict = train(
            args,
            model_tasks,
            model,
            None,
            None,
            loss_fn,
            train_loader,
            epoch,
            device,
            tensor_writer,
        )
        train_loss = np.mean(train_losses)
        losses_dict["train_loss_per_step"].extend(train_losses)
        losses_dict["train_loss_per_epoch"].append(train_loss)

        # Validate
        val_loss, val_metrics_dict = validate(
            args, model_tasks, model, loss_fn, val_loader, epoch, device
        )
        losses_dict["val_loss_per_epoch"].append(val_loss)

        # Adjust learning rate for every epoch instead of step
        # if scheduler is not None:
        #     scheduler.step()
        #     scheduler.step(val_loss)  # applies only when scheduler is ReduceLROnPlateau

        lr = model.get_lr()[0]

        improved = val_loss < best_loss
        if improved:
            best_loss = val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if improved:
            client_state = {'epoch': epoch, 'best_loss': best_loss}
            ckpt_path = os.path.join(checkpoint_save_dir, f"epoch-{epoch}") 
            model.save_checkpoint(ckpt_path, client_state=client_state)
            if is_main_process():
                logger.info(f"[DeepSpeed] train loss: {train_loss}, val loss: {val_loss} \n[DeepSpeed] Model saved: {ckpt_path}")
                num_saved += 1

        if is_main_process():
            logger.info("lr = {:.10f}".format(lr))

            logger.info(f"train loss: {train_loss}, val loss: {val_loss} \nEpochs since last improvement: {epochs_since_improvement}")
            
            # Tensorboard
            if tensor_writer is not None:
                tensor_writer.add_scalars(
                    "train-val.loss/epoch",
                    {"train": train_loss, "val": val_loss},
                    epoch,
                )
                for task in model_tasks:
                    tensor_writer.add_scalars(
                        f"train.{task}.metrics/epoch",
                        train_metrics_dict[task].get_all_metrics(),
                        epoch,
                    )
                    tensor_writer.add_scalars(
                        f"val.{task}.metrics/epoch",
                        val_metrics_dict[task].get_all_metrics(),
                        epoch,
                    )
                    tensor_writer.add_scalars(
                        f"val.{task}.allvalues/epoch",
                        val_metrics_dict[task].to_dict(),
                        epoch,
                    )

            # Save log
            train_metrics_str = "* [Train Metrics]"
            val_metrics_str = "* [Val Metrics]"
            for task in model_tasks:
                train_metrics_str += f"[{task.upper()}]{train_metrics_dict[task]} "
                val_metrics_str += f"[{task.upper()}]{val_metrics_dict[task]} "
            logger.info(train_metrics_str)
            logger.info(val_metrics_str)

            # Time
            epoch_end_time = datetime.datetime.now()
            epoch_cost_time = epoch_end_time - epoch_start_time
            cost_time += epoch_cost_time
            estimated_end_time = (
                (cost_time / (i + 1)) * 0.1 + epoch_cost_time * 0.9
            ) * (args.epochs - (i + 1)) + epoch_end_time
            logger.info(f"* Epoch cost time: {strftimedelta(epoch_cost_time)}")
            logger.info(
                f"* Estimated end time: {estimated_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
        
        # Early stopping
        stop_training = epochs_since_improvement > args.patience
        if is_dist_avail_and_initialized():
            stop_training = broadcast_object(stop_training, src=0)
        if stop_training:
            if is_main_process():
                logger.warning("\n* Stop training.")
            break

    # Save loss as npy
    if is_main_process():
        loss_save_dir = os.path.join(log_dir, "loss")
        if not os.path.exists(loss_save_dir):
            os.makedirs(loss_save_dir)
        for name, t in losses_dict.items():
            if isinstance(t, torch.Tensor):
                t = t.detach().cpu().numpy()
            np.save(os.path.join(loss_save_dir, f"{args.model_name}_{name}.npy"), t)

    # Broadcast to all processes
    if is_dist_avail_and_initialized():
        ckpt_path = broadcast_object(ckpt_path, src=0)

    return ckpt_path