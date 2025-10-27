# File: train_distributed.py
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as torch_mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm
import datetime
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import glob
import re
import shutil
import signal
import sys

from TransNet_train import TransNet, custom_loss

# Global flag to track if the process received a signal
received_signal = False

def signal_handler(signum, frame):
    """Handle termination signals to save checkpoint before exiting"""
    global received_signal
    received_signal = True
    print(f"Received signal {signum}, initiating clean shutdown...")
    logging.warning(f"Received signal {signum}, initiating clean shutdown...")
    # Let the main loop handle the actual checkpoint saving

def verify_distributed_setup(rank, world_size, dataset):
    """Verify that data is correctly partitioned across GPUs"""
    # Initialize distributed environment
    setup_distributed(rank, world_size)
    
    # Create sampler and dataloader
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # Create dataloader with a small batch size for testing
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler
    )
    
    # Get the first few batches and print identifying information
    sampler.set_epoch(0)  # Set epoch to ensure proper shuffling
    
    print(f"Rank {rank}: Sampler has {len(sampler)} samples")
    
    batch_samples = []
    for i, (batch_x, _) in enumerate(dataloader):
        # Extract sample ID from first element of batch
        sample_id = batch_x[0, 0, 0, 0].item()
        batch_samples.append(sample_id)
        
        if i >= 3:  # Check just a few batches
            break
    
    print(f"Rank {rank} processing samples: {batch_samples}")
    
    # Wait for all processes to reach this point
    dist.barrier()
    
    # Clean up
    cleanup()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
            
        if self.mode == 'min':
            delta = self.best_loss - val_loss
        else:
            delta = val_loss - self.best_loss
            
        if delta > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def load_state(self, state_dict):
        """Load state from checkpoint"""
        self.counter = state_dict.get('counter', 0)
        self.best_loss = state_dict.get('best_loss', None)
        self.early_stop = state_dict.get('early_stop', False)
    
    def get_state(self):
        """Get state for saving in checkpoint"""
        return {
            'counter': self.counter,
            'best_loss': self.best_loss,
            'early_stop': self.early_stop
        }

def setup_logging():
    """Set up logging configuration for training"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Configure root logger
    log_file = f'logs/training_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger('ADR-GNN')


def setup_distributed(rank, world_size):
    """Initialize the distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12399'
    # Set longer timeout for NCCL operations to avoid timeouts
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # Give each process enough time to initialize
    time.sleep(rank * 2)
    
    try:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=60)  # Increased timeout
        )
        
        torch.cuda.set_device(rank)
        print(f"Initialized process {rank}/{world_size} on GPU {torch.cuda.current_device()}")
    except Exception as e:
        print(f"Error initializing process group on rank {rank}: {str(e)}")
        raise e


def cleanup():
    """Clean up distributed environment"""
    if dist.is_initialized():
        try:
            dist.barrier()  # Synchronize before destroying
        except:
            pass  # If barrier fails, still try to clean up
        
        try:
            dist.destroy_process_group()
        except:
            pass  # Continue even if destroy fails

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """Find the latest checkpoint in the directory with improved handling for different formats"""
    # First check regular checkpoints with batch information
    checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*_batch_*.pt')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        # Next try regular checkpoints with just epoch information
        checkpoint_pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if not checkpoint_files:
            # Finally check for emergency checkpoints
            emergency_pattern = os.path.join(checkpoint_dir, 'emergency_checkpoint_*.pt')
            checkpoint_files = glob.glob(emergency_pattern)
            
            if not checkpoint_files:
                return None
    
    # Extract epoch and batch numbers
    checkpoints = []
    for file in checkpoint_files:
        # Extract epoch and batch from filename
        if 'emergency_checkpoint' in file:
            # Emergency checkpoints - extract timestamp
            match = re.search(r'emergency_checkpoint_(\d+)\.pt', os.path.basename(file))
            if match:
                timestamp = int(match.group(1))
                # Use high epoch number to prioritize emergency checkpoints
                checkpoints.append((999999, timestamp, file))
        elif '_batch_' in file:
            # Format with both epoch and batch
            match = re.search(r'checkpoint_epoch_(\d+)_batch_(\d+)\.pt', os.path.basename(file))
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                checkpoints.append((epoch, batch, file))
        else:
            # Format with just epoch
            match = re.search(r'checkpoint_epoch_(\d+)\.pt', os.path.basename(file))
            if match:
                epoch = int(match.group(1))
                # Default batch to 0
                checkpoints.append((epoch, 0, file))
    
    if not checkpoints:
        return None
    
    # Sort by epoch (primary) and batch (secondary) in descending order
    checkpoints.sort(reverse=True, key=lambda x: (x[0], x[1]))
    
    # Return the path to the latest checkpoint
    _, _, latest_checkpoint = checkpoints[0]
    print(f"Found latest checkpoint: {latest_checkpoint}")  # Add debug print
    return latest_checkpoint

class DistributedTrainer:
    def __init__(self, rank, world_size, model, train_dataset, val_dataset): # , test_dataset
        """
        Initialize distributed trainer
        
        Args:
            rank: GPU rank for this process
            world_size: Total number of GPUs
            model: ADR-GNN model instance
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
        """
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        self.early_stopping = EarlyStopping(patience=5, min_delta=1e-4, mode='min')
        
        # Create checkpoints directory
        if rank == 0:
            os.makedirs('checkpoints', exist_ok=True)
            # Backup directory for atomic checkpointing
            os.makedirs('checkpoints/tmp', exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger('ADR-GNN')
        if rank == 0:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)  # Changed from ERROR to WARNING
        
        # Training hyperparameters 
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.01
        self.max_grad_norm = 1.0
        
        # Track training state
        self.start_epoch = 0
        self.current_epoch = 0
        self.best_val_mae = float('inf')
        self.best_val_ioa = float('inf')
        self.checkpoint_interval = 1  # Save checkpoint every epoch (increased frequency)
        self.batch_checkpoint_interval = 20  # Save checkpoint every N batches within an epoch
        self.last_batch_checkpoint = 0  # Track last time a batch checkpoint was saved
        
        # Track batch-level progress for resuming
        self.resume_batch_idx = 0
        
        # Optimize GPU memory usage
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
        # Move model to device and wrap with DDP
        self.model = model.to(self.device)
        
        # DDP gradient sync params
        self.find_unused_parameters = True  # Set to True if needed
        
        self.model = DDP(
            self.model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=self.find_unused_parameters,
            broadcast_buffers=False
        )
        
        # Set up loss and optimizer
        self.criterion = custom_loss()
        self.optimizer = torch.optim.Adamax(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.0005,
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=3,
            factor=0.2,
            min_lr=1e-6,
            threshold=1e-4,
        )
        
        # Create data samplers
        self.train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False
        )
        
        self.val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        )
        
        # Create data loaders with smaller num_workers to reduce memory pressure
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=8,  # Reduced from 32 to prevent resource issues
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=8,  # Reduced from 32
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        # Register signal handlers for clean shutdown
        if rank == 0:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        
        # Load checkpoint if available
        self.load_checkpoint()
            
    def load_checkpoint(self):
        """Load the latest checkpoint if available with robust error handling"""
        if self.rank == 0:
            self.logger.info("Checking for existing checkpoints...")
    
        # Find latest checkpoint
        latest_checkpoint = find_latest_checkpoint()
        
        if latest_checkpoint is None:
            if self.rank == 0:
                self.logger.info("No checkpoints found. Starting training from epoch 0.")
            return
        
        if self.rank == 0:
            self.logger.info(f"Found checkpoint: {latest_checkpoint}")
        
        # Broadcast checkpoint path from rank 0 to all processes
        object_list = [latest_checkpoint]
        if self.world_size > 1:
            try:
                dist.broadcast_object_list(object_list, src=0)
                latest_checkpoint = object_list[0]
            except Exception as e:
                self.logger.warning(f"Error broadcasting checkpoint path: {str(e)}. "
                                   f"Each process will load independently.")
        
        # Load checkpoint on all processes
        try:
            # Map checkpoint to the correct device
            map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
            checkpoint = torch.load(latest_checkpoint, map_location=map_location)
            
            # Extract batch index from filename if present (explicit check)
            batch_idx_from_file = 0
            if '_batch_' in latest_checkpoint:
                match = re.search(r'checkpoint_epoch_(\d+)_batch_(\d+)\.pt', os.path.basename(latest_checkpoint))
                if match:
                    batch_idx_from_file = int(match.group(2))
                    if self.rank == 0:
                        self.logger.info(f"Detected batch index from filename: {batch_idx_from_file}")
            
            # Load model state with detailed error reporting
            try:
                missing_keys, unexpected_keys = self.model.module.load_state_dict(
                    checkpoint['model_state_dict'], strict=False
                )
                
                if missing_keys and self.rank == 0:
                    self.logger.warning(f"Missing keys when loading model: {missing_keys}")
                if unexpected_keys and self.rank == 0:
                    self.logger.warning(f"Unexpected keys when loading model: {unexpected_keys}")
            except Exception as e:
                self.logger.error(f"Error loading model state: {str(e)}")
                raise
            
            # Load optimizer state with error handling
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer state to the correct device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            except Exception as e:
                self.logger.error(f"Error loading optimizer state: {str(e)}")
            
            # Load scheduler state with error handling
            if 'scheduler_state_dict' in checkpoint:
                try:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except Exception as e:
                    self.logger.error(f"Error loading scheduler state: {str(e)}")
            
            # Extract training state information - use batch info from filename if needed
            self.start_epoch = checkpoint.get('epoch', 0)
            # Get batch index from checkpoint, fallback to filename extraction
            self.resume_batch_idx = checkpoint.get('batch_idx', batch_idx_from_file)
            
            # Check for completed epoch flag
            completed_epoch = checkpoint.get('completed_epoch', False)
            if completed_epoch:
                # If the epoch was completed, start from the next epoch
                self.start_epoch += 1
                self.resume_batch_idx = 0
                if self.rank == 0:
                    self.logger.info(f"Previous epoch was completed. Starting from epoch {self.start_epoch}")
            else:
                if self.rank == 0:
                    self.logger.info(f"Resuming from epoch {self.start_epoch}, batch {self.resume_batch_idx}")
                
            self.current_epoch = self.start_epoch
            
            # Load best metrics
            if 'best_val_mae' in checkpoint:
                self.best_val_mae = checkpoint['best_val_mae']
            
            if 'best_val_ioa' in checkpoint:
                self.best_val_ioa = checkpoint['best_val_ioa']
            
            # Load early stopping state
            if 'early_stopping' in checkpoint:
                self.early_stopping.load_state(checkpoint['early_stopping'])
            
            if self.rank == 0:
                self.logger.info(f"Best validation MAE: {self.best_val_mae:.4f}")
                self.logger.info(f"Best validation IOA: {self.best_val_ioa:.4f}")
                
        except Exception as e:
            if self.rank == 0:
                self.logger.error(f"Error loading checkpoint: {str(e)}")
                self.logger.info("Starting training from epoch 0.")
            self.start_epoch = 0
            self.current_epoch = 0
            self.resume_batch_idx = 0
            self.best_val_mae = float('inf')
            self.best_val_ioa = float('inf')
        
        # Synchronize processes after checkpoint loading
        if self.world_size > 1:
            try:
                # Make sure all processes have loaded the checkpoint
                dist.barrier()
            except Exception as e:
                self.logger.error(f"Error in barrier after checkpoint loading: {str(e)}")
        
    def safe_save_checkpoint(self, checkpoint_data, filepath):
        """Safe atomic checkpoint saving to prevent corruption"""
        if self.rank != 0:
            return False
        
        # Save to a temporary file first
        tmp_path = os.path.join('checkpoints/tmp', os.path.basename(filepath))
        try:
            torch.save(checkpoint_data, tmp_path)
            # Move the file to the final destination (atomic operation on most filesystems)
            shutil.move(tmp_path, filepath)
            return True
        except Exception as e:
            self.logger.error(f"Error saving checkpoint to {filepath}: {str(e)}")
            return False
    
    def save_checkpoint(self, epoch, batch_idx=0, is_best=False, is_emergency=False, completed_epoch=False):
        """Save training checkpoint with completed_epoch flag"""
        if self.rank != 0:
            return
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs('checkpoints/tmp', exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'batch_idx': batch_idx,
            'completed_epoch': completed_epoch,  # Add flag to indicate if epoch is fully complete
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_mae': self.best_val_mae,
            'best_val_ioa': self.best_val_ioa,
            'early_stopping': self.early_stopping.get_state()
        }
        
        # Save regular or emergency checkpoint
        if is_emergency:
            timestamp = int(time.time())
            checkpoint_path = f'checkpoints/emergency_checkpoint_{timestamp}.pt'
            success = self.safe_save_checkpoint(checkpoint, checkpoint_path)
            if success:
                self.logger.info(f"Saved emergency checkpoint at epoch {epoch}, batch {batch_idx} to {checkpoint_path}")
        else:
            if batch_idx > 0:
                # Save with batch information
                checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}_batch_{batch_idx}.pt'
            else:
                # Save with just epoch information
                checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}.pt'
                
            success = self.safe_save_checkpoint(checkpoint, checkpoint_path)
            if success:
                if completed_epoch:
                    self.logger.info(f"Saved checkpoint for completed epoch {epoch} to {checkpoint_path}")
                else:
                    self.logger.info(f"Saved checkpoint at epoch {epoch}, batch {batch_idx} to {checkpoint_path}")
        
        # Save best model if this is the best so far
        if is_best:
            best_path = 'checkpoints/best_model.pt'
            success = self.safe_save_checkpoint(checkpoint, best_path)
            if success:
                self.logger.info(f"Saved best model to {best_path}")
        
        # Optionally, remove older checkpoints to save disk space
        self.cleanup_old_checkpoints(keep=5)
    
    def save_batch_checkpoint(self, epoch, batch_idx):
        """Save checkpoint at batch level for finer-grained resuming"""
        # Only save if enough batches have passed since last save
        current_batch = batch_idx
        if current_batch - self.last_batch_checkpoint >= self.batch_checkpoint_interval:
            self.save_checkpoint(epoch, batch_idx=batch_idx)
            self.last_batch_checkpoint = current_batch
    
    def cleanup_old_checkpoints(self, keep=5):
        """Remove old checkpoints, keeping only the latest 'keep' ones and the best model"""
        checkpoint_pattern = os.path.join('checkpoints', 'checkpoint_epoch_*_batch_*.pt')
        checkpoint_files = glob.glob(checkpoint_pattern)
        
        if len(checkpoint_files) <= keep:
            return
        
        # Extract epoch and batch numbers and sort checkpoints
        checkpoints = []
        for file in checkpoint_files:
            match = re.search(r'checkpoint_epoch_(\d+)_batch_(\d+)\.pt', file)
            if match:
                epoch = int(match.group(1))
                batch = int(match.group(2))
                checkpoints.append((epoch, batch, file))
        
        # Sort by epoch and batch (descending)
        checkpoints.sort(reverse=True, key=lambda x: (x[0], x[1]))
        
        # Keep the best and the latest 'keep' checkpoints
        keep_files = set([cp[2] for cp in checkpoints[:keep]])
        best_model = os.path.join('checkpoints', 'best_model.pt')
        if os.path.exists(best_model):
            keep_files.add(best_model)
        
        # Don't delete emergency checkpoints
        emergency_pattern = os.path.join('checkpoints', 'emergency_checkpoint_*.pt')
        emergency_files = glob.glob(emergency_pattern)
        for file in emergency_files:
            keep_files.add(file)
        
        # Remove old checkpoints
        for _, _, file in checkpoints[keep:]:
            if file not in keep_files:
                try:
                    os.remove(file)
                    self.logger.debug(f"Removed old checkpoint: {file}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove old checkpoint {file}: {str(e)}")
        
    def calculate_metrics(self, outputs, targets):
        """Calculate RMSE, MAE and IOA metrics"""
        with torch.no_grad():
            # Calculate MSE and RMSE
            mse = torch.mean((outputs - targets) ** 2)
            rmse = torch.sqrt(mse)
            mae = torch.mean(torch.abs(outputs - targets))
            target_mean = torch.mean(targets)
            numerator = torch.sum((outputs - targets) ** 2)
            denominator = torch.sum((torch.abs(outputs - target_mean) + 
                                   torch.abs(targets - target_mean)) ** 2)
            ioa = 1 - (numerator / (denominator + 1e-6))
            
            return rmse.item(), mae.item(), ioa.item()
        
    def train_epoch(self, epoch):
        """Train for one epoch with robust error handling and checkpointing"""
        self.model.train()
        total_mae = 0.0
        total_ioa = 0.0
        processed_batches = 0
        
        # Set epoch for sampler shuffling
        self.train_loader.sampler.set_epoch(epoch)
        
        # Create an iterator so we can manually control the batches
        train_iterator = iter(self.train_loader)
        num_batches = len(self.train_loader)
        
        # Start from the saved batch if resuming
        start_batch = 0
        if epoch == self.start_epoch and self.resume_batch_idx > 0:
            start_batch = self.resume_batch_idx
            if self.rank == 0:
                self.logger.info(f"Explicitly skipping to batch {start_batch}/{num_batches} for resumption")

            try:
                # self.logger.info(f"Skipping to batch {start_batch}/{num_batches} for resumption")
                for _ in range(start_batch):
                    next(train_iterator)
                if self.rank == 0:
                    self.logger.info(f"Successfully skipped to batch {start_batch}")
            
            except StopIteration:
                self.logger.error(f"Failed to skip to batch {start_batch} - not enough batches")
                return 0.0, 0.0
        
        if self.rank == 0:
            pbar = tqdm(total=num_batches, initial=start_batch, desc=f'Epoch {epoch}')
        
        # Process batches
        for batch_idx in range(start_batch, num_batches):
            # Check if we received a termination signal
            global received_signal
            if received_signal and self.rank == 0:
                self.logger.warning("Received termination signal, saving checkpoint...")
                self.save_checkpoint(epoch, batch_idx=batch_idx, is_emergency=True)
                self.logger.warning("Emergency checkpoint saved, exiting...")
                sys.exit(0)
            
            try:
                # Get next batch
                batch_x, batch_y = next(train_iterator)
                
                # Move data to GPU
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                # Forward pass
                self.optimizer.zero_grad()
                try:
                    outputs = self.model(batch_x)
                    mae, ioa = self.criterion(outputs, batch_y)
                    
                    # Backward pass
                    ioa.backward()   # ioa is negative of actual IOA
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if self.rank == 0:
                            self.logger.error(f"GPU OOM error at batch {batch_idx}: {str(e)}")
                            # Save emergency checkpoint
                            self.logger.warning("Saving emergency checkpoint...")
                            self.save_checkpoint(epoch, batch_idx=batch_idx, is_emergency=True)
                        torch.cuda.empty_cache()
                        # Skip this batch and continue
                        continue
                    else:
                        raise
                
                # Calculate and apply gradient clipping
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                except RuntimeError as e:
                    self.logger.error(f"Error during gradient clipping: {str(e)}")
                    continue
                
                # Synchronize gradients - with error handling
                try:
                    # Only synchronize if we haven't set find_unused_parameters=True in DDP
                    if not self.find_unused_parameters:
                        for param in self.model.parameters():
                            if param.grad is not None:
                                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                                param.grad.data /= self.world_size
                except RuntimeError as e:
                    if self.rank == 0:
                        self.logger.error(f"Error during gradient synchronization: {str(e)}")
                        # Save emergency checkpoint
                        self.save_checkpoint(epoch, batch_idx=batch_idx, is_emergency=True)
                    # Try to recover
                    torch.cuda.empty_cache()
                    continue
                
                # Update parameters
                self.optimizer.step()
                
                # Update metrics
                total_mae += mae.item()
                total_ioa += ioa.item()
                processed_batches += 1
                
                # Update progress bar on master process
                if self.rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    pbar.set_postfix({
                        'MAE': f"{total_mae/processed_batches:.4f}",
                        'IOA': f"{total_ioa/processed_batches:.4f}",
                        'lr': f"{current_lr:.6f}"
                    })
                    pbar.update(1)
                
                # Save periodic batch checkpoints
                if self.rank == 0 and batch_idx % self.batch_checkpoint_interval == 0 and batch_idx > 0:
                    self.save_batch_checkpoint(epoch, batch_idx)
                    
            except StopIteration:
                break
            except Exception as e:
                if self.rank == 0:
                    self.logger.error(f"Error in batch {batch_idx}: {str(e)}")
                    self.save_checkpoint(epoch, batch_idx=batch_idx, is_emergency=True)
                # Try to continue with next batch
                continue
        
        if self.rank == 0:
            self.save_checkpoint(epoch, batch_idx=0, completed_epoch=True)# pbar.close()
        
        # Compute average metrics
        avg_mae = total_mae / max(processed_batches, 1)  # Avoid division by zero
        avg_ioa = total_ioa / max(processed_batches, 1)
        
        # Synchronize average metrics across processes
        if processed_batches > 0:
            metrics_tensor = torch.tensor([avg_mae, avg_ioa, float(processed_batches)], device=self.device)
            try:
                dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
                # Weighted average based on processed batches
                if metrics_tensor[2].item() > 0:  # Total processed batches across all ranks
                    avg_mae = metrics_tensor[0].item() / self.world_size
                    avg_ioa = metrics_tensor[1].item() / self.world_size
            except Exception as e:
                self.logger.error(f"Error synchronizing metrics: {str(e)}")
        
        return avg_mae, avg_ioa
    
    def validate(self):
        """Perform validation with error handling"""
        self.model.eval()
        total_mae = 0.0
        total_ioa = 0.0
        processed_batches = 0
        
        try:
            with torch.no_grad():
                for batch_x, batch_y in self.val_loader:
                    try:
                        batch_x = batch_x.to(self.device, non_blocking=True)
                        batch_y = batch_y.to(self.device, non_blocking=True)
                        
                        outputs = self.model(batch_x)
                        mae, ioa = self.criterion(outputs, batch_y)
                        total_mae += mae.item()
                        total_ioa += ioa.item()
                        processed_batches += 1
                    except Exception as e:
                        self.logger.error(f"Error in validation batch: {str(e)}")
                        continue
            
            # Compute average metrics
            avg_mae = total_mae / max(processed_batches, 1)
            avg_ioa = total_ioa / max(processed_batches, 1)
            
            # All-reduce validation metrics with error handling
            metrics = torch.tensor([avg_mae, avg_ioa, float(processed_batches)], device=self.device)
            try:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
                # Weighted average based on processed batches
                if metrics[2].item() > 0:  # Total processed batches
                    avg_mae = metrics[0].item() / self.world_size
                    avg_ioa = metrics[1].item() / self.world_size
            except Exception as e:
                self.logger.error(f"Error synchronizing validation metrics: {str(e)}")
            
            return avg_mae, avg_ioa
            
        except Exception as e:
            self.logger.error(f"Error during validation: {str(e)}")
            # Return previous values if validation fails completely
            return float('inf'), float('inf')
    
    def train(self):
        """Main training loop with robust error handling and checkpointing"""
        try:
            for epoch in range(self.start_epoch, self.num_epochs):
                self.current_epoch = epoch
                
                # Reset batch checkpoint counter for new epoch
                self.last_batch_checkpoint = 0
                self.resume_batch_idx = 0
                
                # Training
                train_mae, train_ioa = self.train_epoch(epoch)

                if self.rank == 0:
                    self.logger.info(f'Epoch {epoch} training completed. Starting validation...')
                
                # Save epoch checkpoint after training
                if self.rank == 0:
                    self.save_checkpoint(epoch, batch_idx=0)
                
                # Validation
                val_mae, val_ioa = self.validate()
                val_loss = val_mae

                if self.rank == 0:
                    self.logger.info(f'Validation completed. MAE: {val_mar:.4f}, IOA: {val_ioa:.4f}')

                if self.rank == 0:
                    self.logger.info('Updating learning rate scheduler...')
                
                # Update learning rate scheduler
                self.scheduler.step(val_loss)
                
                if self.rank == 0:
                    self.logger.info(f'Epoch {epoch}:')
                    self.logger.info(f'Train MAE: {train_mae:.4f}, Train IOA: {train_ioa:.4f}')
                    self.logger.info(f'Val MAE: {val_mae:.4f}, Val IOA: {val_ioa:.4f}')
                    
                    # Save checkpoint if either metric improves
                    save_model = False
                    if val_loss < self.best_val_mae:
                        self.best_val_mae = val_loss
                        save_model = True
                        self.logger.info(f'New best MAE: {self.best_val_mae:.4f}')
                    
                    if val_ioa < self.best_val_ioa:
                        self.best_val_ioa = val_ioa
                        save_model = True
                        self.logger.info(f'New best IOA: {self.best_val_ioa:.4f}')
                    
                    if save_model:
                        self.save_checkpoint(epoch, is_best=True)
                
                # Check early stopping
                if self.early_stopping(val_loss):
                    if self.rank == 0:
                        self.logger.info(f'Early stopping triggered after epoch {epoch}')
                    break
                
                # Synchronize processes before starting next epoch
                try:
                    dist.barrier()
                except Exception as e:
                    self.logger.error(f"Error in barrier after epoch {epoch}: {str(e)}")
                    # Save emergency checkpoint if barrier fails
                    if self.rank == 0:
                        self.save_checkpoint(epoch, is_emergency=True)
                    # Continue anyway
            
            if self.rank == 0:
                self.logger.info('Training completed!')
            
        except Exception as e:
            self.logger.error(f'Error in process {self.rank}: {str(e)}')
            
            # Save emergency checkpoint if process 0
            if self.rank == 0:
                self.logger.info('Saving emergency checkpoint due to error...')
                self.save_checkpoint(self.current_epoch, is_emergency=True)
                
            raise e
            
        finally:
            # Save final checkpoint
            if self.rank == 0:
                self.logger.info('Saving final checkpoint...')
                self.save_checkpoint(self.current_epoch, is_best=False)
            
            cleanup()

def train_distributed(rank, world_size, model, train_dataset, val_dataset): # , test_dataset
    """Entry point for each distributed process"""
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Initialize distributed environment
        setup_distributed(rank, world_size)

        if rank == 0:
            print(f"Process group initialized with world_size={world_size}")
        
        # Create trainer and start training
        trainer = DistributedTrainer(
            rank,
            world_size, 
            model,
            train_dataset,
            val_dataset,
            # test_dataset
        )
        trainer.train()
        
    except Exception as e:
        logging.error(f'Error in process {rank}: {str(e)}')
        raise e
    
    finally:
        cleanup()

def create_sequences(data, seq_length = 24, target_horizons = 72):
    """
    Create sequences for training with specific forecast horizon
    
    Args:
        data: Shape (samples, nodes, features)
        original_pm25: Shape (samples, nodes) - contains original PM2.5 values
        seq_length: Input sequence length
        target_horizons: List of specific forecast horizons
        
    Returns:
        sequences: Shape (num_sequences, seq_length, nodes, features)
        targets: Shape (num_sequences, len(target_horizons), nodes)
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length - target_horizons):
        # Input sequence
        seq = data[i:(i + seq_length)]
        
        # Target sequence (PM2.5 values only) , original_pm25
        target = [
         data[(i + seq_length) : (i + seq_length + target_horizons), :, -1]]
                
        sequences.append(seq)
        targets.append(target)
        
    return np.array(sequences), np.array(np.squeeze(targets))

def load_data():
    """
    Load preprocessed data from saved numpy files
    
    Returns:
        data: Preprocessed sensor data with shape (timesteps, stations, features)
        lats: Latitude coordinates for each station
        lons: Longitude coordinates for each station
    """
    try:
        # Load the preprocessed data files
        # imputed_data = np.load('/pscratch/sd/r/rdimri/imputed_data.npy')
        scaled_data = np.load('/pscratch/sd/r/rdimri/BSL_PM25/scaled_data.npy')
        final_features = np.load('/pscratch/sd/r/rdimri/BSL_PM25/final_features.npy')
        data = np.concatenate([final_features, scaled_data[:-1, :, -1:]], axis = -1)
        lats = np.load('data/lats.npy')
        lons = np.load('data/lons.npy')
        
        if data is None or lats is None or lons is None:
            raise ValueError("One or more data files could not be loaded")
            
        return data, lats, lons
        
    except FileNotFoundError as e:
        logging.error(f"Could not find required data files: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def process_data(data):
    """
    Split data into training and validation sets
    
    Args:
        data: Full dataset with shape (timesteps, stations, features)
        original_pm25: Original PM2.5 values with shape (timesteps, stations)
    Returns:
        train_data: Training data (first 2 years, 2018 and 2019)
        val_data: Validation data (next 1 year, 2020)
        test_data: Test data (last year, 2021)
    """
    # Split at 2-year mark (2*8760 hours)
    train_data = data[:2*8760]
    val_data = data[2*8760:3*8760]
    test_data = data[3*8760:]
    
    logging.info(f"Training data shape: {train_data.shape}")
    logging.info(f"Validation data shape: {val_data.shape}")
    logging.info(f"Test data shape: {test_data.shape}")
    
    return train_data, val_data, test_data

def main():
    """Main entry point for training"""
    # Set up logging first
    logger = setup_logging()
    logger.info("Starting ADR-GNN training")
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('checkpoints/tmp', exist_ok=True)
    
    # Register signal handlers for main process
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Load and process data
        logger.info("Loading data...")
        data, lats, lons = load_data()
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Split into train/val
        logger.info("Processing data splits...")
        train_data, val_data, test_data = process_data(data)
        
        # Create sequences_optimized
        logger.info("Creating training sequences...")
        X_train, y_train = create_sequences(train_data, seq_length=24, target_horizons=72)
        X_val, y_val = create_sequences(val_data, seq_length=24, target_horizons=72)
        
        logger.info(f"Training sequences shape: {X_train.shape}")
        logger.info(f"Training targets shape: {y_train.shape}")
        
        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        # Initialize model
        model = ADR_GNN(
            input_features=data.shape[-1],
            n_stations=len(lats),
            seq_length=24,
            pred_length=72,
            L=1,
            state_dim=64,
            hist_dim=64,
        )
        
        # Initialize graph processor
        logger.info("Initializing graph processor...")
        model.initialize_graph_processor(lats, lons)
        
        # Set random seeds
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        # Launch distributed training
        world_size = torch.cuda.device_count()
        
        logger.info(f"Starting distributed training with {world_size} GPUs")
        
        try:
            torch_mp.spawn(
                train_distributed,
                args=(world_size, model, train_dataset, val_dataset),
                nprocs=world_size,
                join=True
            )
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise e
            
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        raise e

if __name__ == '__main__':
    main()
