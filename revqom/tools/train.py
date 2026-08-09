# adapted from opencood, ucla/v2x-real
# Author: Dereje Shenkut <dshenkut@andrew.cmu.edu>


import argparse
import os
import statistics
import gc
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import revqom.hypes_yaml.yaml_utils as yaml_utils
from revqom.tools import train_utils
from revqom.tools import multi_gpu_utils
from revqom.data_utils.datasets import build_dataset
from revqom.tools import train_utils


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true',
                        help='whether to use distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    # Initialize distributed mode
    multi_gpu_utils.init_distributed_mode(opt)
    torch.manual_seed(2025)  # You can choose any seed value
    
    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)
    # breakpoint()
    # # Create a small subset of scenario_folders for both datasets
    # opencood_train_dataset.scenario_folders = opencood_train_dataset.scenario_folders[:4]
    # opencood_validate_dataset.scenario_folders = opencood_validate_dataset.scenario_folders[:2]
    
    # # Reinitialize datasets with new scenario folders
    # opencood_train_dataset.reinitialize()
    # opencood_validate_dataset.reinitialize()

    # Setup distributed sampling
    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset, shuffle=True)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            sampler=sampler_val,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            drop_last=False
        )
    else:
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=hypes['train_params']['batch_size'],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        val_loader = DataLoader(
            opencood_validate_dataset,
            batch_size=hypes['train_params']['batch_size'],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=False,
            pin_memory=True,
            drop_last=True
        )
    
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check retrofit mode early
    retrofit_mode = hypes.get('retrofit_mode', False)
    
    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # Move model to GPU first
    if torch.cuda.is_available():
        model.to(device)
    
    # Handle retrofit mode AFTER moving to device
    if retrofit_mode:
        pretrained_path = hypes.get('pretrained_model_path')
        if pretrained_path:
            # Check if model has EigenMAP retrofit methods
            if hasattr(model, 'load_pretrained_for_revqom_retrofit'):
                # EigenMAP retrofit training
                model.load_pretrained_for_revqom_retrofit(pretrained_path)
                print(f"EigenMAP Retrofit mode: Loaded pretrained model from {pretrained_path}")
            elif hasattr(model, 'load_pretrained_for_retrofit'):
                # General retrofit training
                model.load_pretrained_for_retrofit(pretrained_path)
                model.freeze_for_retrofit()
                print(f"General Retrofit mode: Loaded pretrained model from {pretrained_path}")
            else:
                print("Warning: Model does not support retrofit training")
        else:
            print("Warning: Retrofit mode enabled but no pretrained_model_path specified")
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module
    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup - special handling for retrofit mode
    if retrofit_mode:
        # Only optimize unfrozen parameters (codec only)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Retrofit mode: Training {len(trainable_params)} parameter groups (codec only)")
        
        # Simple Adam optimizer for retrofit
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=hypes['optimization']['lr'],
            weight_decay=hypes['optimization']['weight_decay'],
            betas=tuple(hypes['optimization'].get('betas', [0.9, 0.999]))
        )
        onecycle_scheduler = None  # No complex scheduler for retrofit
    else:
        # Normal optimizer setup for regular training
        optimizer, onecycle_scheduler = train_utils.setup_optimizer(hypes, model)
    
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    
    # Get warmup settings for retrofit
    warmup_epochs = hypes.get('train_params', {}).get('warmup_epochs', 0)
    target_nq = hypes.get('train_params', {}).get('target_nq', 3)
    warmup_nq = hypes.get('train_params', {}).get('warmup_nq', 1)
    
    for epoch in range(init_epoch, max(epoches, init_epoch)):
        # Clear memory before each epoch
        gc.collect()
        torch.cuda.empty_cache()
        
        # Handle n_q warmup for retrofit mode
        if retrofit_mode and epoch == warmup_epochs and warmup_epochs > 0:
            print(f"Warmup complete at epoch {epoch}. Updating n_q from {warmup_nq} to {target_nq}")
            model_without_ddp.update_nq_warmup(target_nq)
            # Update optimizer for new parameters (codec only)
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(
                trainable_params,
                lr=hypes['optimization']['lr'] * 0.5,  # Lower LR after warmup
                weight_decay=hypes['optimization']['weight_decay'],
                betas=tuple(hypes['optimization'].get('betas', [0.9, 0.999]))
            )
        
        # Update model epoch for curriculum learning
        if hasattr(model_without_ddp, 'set_epoch'):
            model_without_ddp.set_epoch(epoch)
        
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        
        total_loss = 0
        iou_list = []
        epoch_compression_losses = []
        epoch_reconstruction_losses = []
        epoch_vq_losses = []
        epoch_ortho_losses = []
        epoch_perplexities = []
        pbar = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)
            # breakpoint()
            # Handle half precision training
            if not opt.half:
                output_dict = model(batch_data['ego'])
                final_loss, loss_dict = criterion(output_dict, batch_data['ego']['gt_boxes'])
            else:
                with torch.cuda.amp.autocast():
                    output_dict = model(batch_data['ego'])
                    final_loss, loss_dict = criterion(output_dict, batch_data['ego']['gt_boxes'])
            
            # Add EigenMAP compression loss if available
            if 'compression_stats' in output_dict:
                compression_stats = output_dict['compression_stats']
                
                # Add reconstruction loss for EigenMAP (weighted by small factor)
                if 'reconstruction_error' in compression_stats:
                    revqom_recon_loss = compression_stats['reconstruction_error'] * 0.01  # Small weight
                    final_loss = final_loss + revqom_recon_loss
                    loss_dict['revqom_recon_loss'] = revqom_recon_loss
                
                # Log compression statistics
                if i % 20 == 0:  # Log every 20 iterations
                    stats_str = f"EigenMAP - Compression: {compression_stats.get('actual_compression_ratio', 0):.1f}x, "
                    stats_str += f"Rank: {compression_stats.get('avg_rank', 0):.1f}, "
                    stats_str += f"Recon Error: {compression_stats.get('reconstruction_error', 0):.6f}, "
                    stats_str += f"Bits/pixel: {compression_stats.get('bits_per_pixel', 0):.1f}"
                    print(stats_str)
            
            # Loss logging
            total_loss += final_loss.item()
            
            # Log VQ and ortho losses from model output
            if 'vq_loss' in output_dict and output_dict['vq_loss'] is not None:
                vq_loss_val = output_dict['vq_loss'].item() if torch.is_tensor(output_dict['vq_loss']) else output_dict['vq_loss']
                epoch_vq_losses.append(vq_loss_val)
                writer.add_scalar('Train/VQLoss', vq_loss_val, epoch * len(train_loader) + i)
            
            if 'ortho_loss' in output_dict and output_dict['ortho_loss'] is not None:
                ortho_loss_val = output_dict['ortho_loss'].item() if torch.is_tensor(output_dict['ortho_loss']) else output_dict['ortho_loss']
                epoch_ortho_losses.append(ortho_loss_val)
                writer.add_scalar('Train/OrthoLoss', ortho_loss_val, epoch * len(train_loader) + i)
                
            if 'rvq_perplexity' in output_dict and output_dict['rvq_perplexity'] is not None:
                perplexity_val = output_dict['rvq_perplexity'].item() if torch.is_tensor(output_dict['rvq_perplexity']) else output_dict['rvq_perplexity']
                epoch_perplexities.append(perplexity_val)
                writer.add_scalar('Train/RVQPerplexity', perplexity_val, epoch * len(train_loader) + i)
            
            # Log compression losses if available
            if hasattr(criterion, 'loss_dict') and criterion.loss_dict:
                if 'compression_loss' in criterion.loss_dict:
                    compression_loss_val = criterion.loss_dict['compression_loss']
                    epoch_compression_losses.append(compression_loss_val)
                    writer.add_scalar('Train/CompressionLoss', compression_loss_val, epoch * len(train_loader) + i)
                if 'reconstruction_loss' in criterion.loss_dict:
                    reconstruction_loss_val = criterion.loss_dict['reconstruction_loss']
                    epoch_reconstruction_losses.append(reconstruction_loss_val)
                    writer.add_scalar('Train/ReconstructionLoss', reconstruction_loss_val, epoch * len(train_loader) + i)
                # Log the new recon_loss from SVD compression
                if 'recon_loss' in criterion.loss_dict:
                    recon_loss_val = criterion.loss_dict['recon_loss']
                    epoch_reconstruction_losses.append(recon_loss_val)
                    writer.add_scalar('Train/ReconLoss', recon_loss_val, epoch * len(train_loader) + i)
                if 'weighted_recon_loss' in criterion.loss_dict:
                    weighted_recon_loss_val = criterion.loss_dict['weighted_recon_loss']
                    writer.add_scalar('Train/WeightedReconLoss', weighted_recon_loss_val, epoch * len(train_loader) + i)
                # Log EigenMAP compression loss
                if 'revqom_recon_loss' in loss_dict:
                    revqom_recon_loss_val = loss_dict['revqom_recon_loss']
                    writer.add_scalar('Train/EigenMAPReconLoss', revqom_recon_loss_val, epoch * len(train_loader) + i)
            
            if hasattr(criterion, 'iou') and i%10 == 0:
                iou_list.append(criterion.iou)
                # Print per-class IoU if available
                if hasattr(criterion, 'per_class_iou') and criterion.per_class_iou:
                    class_names = opencood_train_dataset.class_names  # ['vehicle', 'pedestrian', 'truck']
                    if not isinstance(class_names, list):
                        class_names = list(class_names)
                    iou_str = " | ".join([f"{class_names[i]}: {criterion.per_class_iou.get(i, 0.0):.3f}" 
                                        for i in range(len(class_names))])
                    print(f"IoU: {criterion.iou:.3f} | Per-class: {iou_str}")
                else:
                    print(f"IoU: {criterion.iou}")
            
            # Backward pass with gradient clipping
            if not opt.half:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

            # Periodic memory cleanup
            if i % 20 == 0:
                torch.cuda.empty_cache()
                
            pbar.update(1)

        # Save model checkpoint
        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                      os.path.join(saved_path, f'net_epoch{epoch + 1}.pth'))

        # Log epoch statistics
        avg_loss = total_loss / len(train_loader)
        avg_iou = sum(iou_list) / len(iou_list) if iou_list else 0
        avg_compression_loss = sum(epoch_compression_losses) / len(epoch_compression_losses) if epoch_compression_losses else 0
        avg_reconstruction_loss = sum(epoch_reconstruction_losses) / len(epoch_reconstruction_losses) if epoch_reconstruction_losses else 0
        avg_vq_loss = sum(epoch_vq_losses) / len(epoch_vq_losses) if epoch_vq_losses else 0
        avg_ortho_loss = sum(epoch_ortho_losses) / len(epoch_ortho_losses) if epoch_ortho_losses else 0
        avg_perplexity = sum(epoch_perplexities) / len(epoch_perplexities) if epoch_perplexities else 0
        
        # Print overall stats
        print(f'\nEpoch {epoch}: Loss = {avg_loss:.4f}, IoU = {avg_iou:.4f}')
        if avg_vq_loss > 0 or avg_ortho_loss > 0:
            print(f'Training - VQ Loss: {avg_vq_loss:.4f}, Ortho Loss: {avg_ortho_loss:.4f}')
        if avg_perplexity > 0:
            print(f'Training - RVQ Perplexity: {avg_perplexity:.4f}')
        if avg_compression_loss > 0:
            print(f'Training - Compression Loss: {avg_compression_loss:.4f}, Reconstruction Loss: {avg_reconstruction_loss:.4f}')
        
        # Print per-class IoU summary if available
        if hasattr(criterion, 'per_class_iou') and criterion.per_class_iou:
            class_names = opencood_train_dataset.class_names
            if not isinstance(class_names, list):
                class_names = list(class_names)
            print(f'Per-class IoU: ', end='')
            for i in range(len(class_names)):
                class_iou = criterion.per_class_iou.get(i, 0.0)
                print(f'{class_names[i]}: {class_iou:.3f}', end=' | ')
                writer.add_scalar(f'Train/IoU_{class_names[i]}', class_iou, epoch)
            print()  # New line
        
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/IoU', avg_iou, epoch)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Log epoch-level VQ and ortho metrics
        if avg_vq_loss > 0:
            writer.add_scalar('Train/EpochVQLoss', avg_vq_loss, epoch)
        if avg_ortho_loss > 0:
            writer.add_scalar('Train/EpochOrthoLoss', avg_ortho_loss, epoch)
        if avg_perplexity > 0:
            writer.add_scalar('Train/EpochRVQPerplexity', avg_perplexity, epoch)
            
        # Log epoch-level compression metrics
        if avg_compression_loss > 0:
            writer.add_scalar('Train/EpochCompressionLoss', avg_compression_loss, epoch)
            writer.add_scalar('Train/EpochReconstructionLoss', avg_reconstruction_loss, epoch)

        # Validation loop
        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_losses = []
            valid_ious = []
            valid_compression_losses = []
            valid_reconstruction_losses = []
            valid_vq_losses = []
            valid_ortho_losses = []
            valid_perplexities = []

            with torch.no_grad():
                for batch_data in val_loader:
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    
                    # Enable compression for validation - use configured compression ratio
                    if 'compression' in hypes['model']['args']:
                        compression_config = hypes['model']['args']['compression']
                        if isinstance(compression_config, dict) and 'args' in compression_config:
                            batch_data['ego']['compress_ratio'] = compression_config['args'].get('ratio', 4)
                        elif isinstance(compression_config, (int, float)) and compression_config > 0:
                            # Backward compatibility for old format
                            batch_data['ego']['compress_ratio'] = compression_config
                    
                    output_dict = model(batch_data['ego'])
                    final_loss, loss_dict = criterion(output_dict, batch_data['ego']['gt_boxes'])
                    
                    # Add EigenMAP compression loss if available (validation)
                    if 'compression_stats' in output_dict:
                        compression_stats = output_dict['compression_stats']
                        if 'reconstruction_error' in compression_stats:
                            revqom_recon_loss = compression_stats['reconstruction_error'] * 0.01
                            final_loss = final_loss + revqom_recon_loss
                    
                    valid_losses.append(final_loss.item())
                    
                    # Collect VQ and ortho losses for validation
                    if 'vq_loss' in output_dict and output_dict['vq_loss'] is not None:
                        vq_loss_val = output_dict['vq_loss'].item() if torch.is_tensor(output_dict['vq_loss']) else output_dict['vq_loss']
                        valid_vq_losses.append(vq_loss_val)
                    
                    if 'ortho_loss' in output_dict and output_dict['ortho_loss'] is not None:
                        ortho_loss_val = output_dict['ortho_loss'].item() if torch.is_tensor(output_dict['ortho_loss']) else output_dict['ortho_loss']
                        valid_ortho_losses.append(ortho_loss_val)
                        
                    if 'rvq_perplexity' in output_dict and output_dict['rvq_perplexity'] is not None:
                        perplexity_val = output_dict['rvq_perplexity'].item() if torch.is_tensor(output_dict['rvq_perplexity']) else output_dict['rvq_perplexity']
                        valid_perplexities.append(perplexity_val)
                    
                    # Collect compression losses for validation
                    if hasattr(criterion, 'loss_dict') and criterion.loss_dict:
                        if 'compression_loss' in criterion.loss_dict:
                            valid_compression_losses.append(criterion.loss_dict['compression_loss'])
                        if 'reconstruction_loss' in criterion.loss_dict:
                            valid_reconstruction_losses.append(criterion.loss_dict['reconstruction_loss'])
                        # Collect the new recon_loss from SVD compression
                        if 'recon_loss' in criterion.loss_dict:
                            valid_reconstruction_losses.append(criterion.loss_dict['recon_loss'])
                    
                    if hasattr(criterion, 'iou'):
                        valid_ious.append(criterion.iou.item() if torch.is_tensor(criterion.iou) else criterion.iou)

            avg_valid_loss = statistics.mean(valid_losses)
            avg_valid_iou = statistics.mean(valid_ious) if valid_ious else 0
            avg_valid_compression_loss = statistics.mean(valid_compression_losses) if valid_compression_losses else 0
            avg_valid_reconstruction_loss = statistics.mean(valid_reconstruction_losses) if valid_reconstruction_losses else 0
            avg_valid_vq_loss = statistics.mean(valid_vq_losses) if valid_vq_losses else 0
            avg_valid_ortho_loss = statistics.mean(valid_ortho_losses) if valid_ortho_losses else 0
            avg_valid_perplexity = statistics.mean(valid_perplexities) if valid_perplexities else 0
            
            print(f'Validation - Loss: {avg_valid_loss:.4f}, IoU: {avg_valid_iou:.4f}')
            if avg_valid_vq_loss > 0 or avg_valid_ortho_loss > 0:
                print(f'Validation - VQ Loss: {avg_valid_vq_loss:.4f}, Ortho Loss: {avg_valid_ortho_loss:.4f}')
            if avg_valid_perplexity > 0:
                print(f'Validation - RVQ Perplexity: {avg_valid_perplexity:.4f}')
            if avg_valid_compression_loss > 0:
                print(f'Validation - Compression Loss: {avg_valid_compression_loss:.4f}, Reconstruction Loss: {avg_valid_reconstruction_loss:.4f}')
            
            writer.add_scalar('Validate/Loss', avg_valid_loss, epoch)
            writer.add_scalar('Validate/IoU', avg_valid_iou, epoch)
            if avg_valid_vq_loss > 0:
                writer.add_scalar('Validate/VQLoss', avg_valid_vq_loss, epoch)
            if avg_valid_ortho_loss > 0:
                writer.add_scalar('Validate/OrthoLoss', avg_valid_ortho_loss, epoch)
            if avg_valid_perplexity > 0:
                writer.add_scalar('Validate/RVQPerplexity', avg_valid_perplexity, epoch)
            if avg_valid_compression_loss > 0:
                writer.add_scalar('Validate/CompressionLoss', avg_valid_compression_loss, epoch)
                writer.add_scalar('Validate/ReconstructionLoss', avg_valid_reconstruction_loss, epoch)

        opencood_train_dataset.reinitialize()       

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
