# VERSION: 2024-10-09
# MODULE | `ModelWrapper` | simplified + freeze-u switch – updated
#
# Adds Option (2): switch inside `fit()` to freeze the surface_fitter (u-MLP)
# and continue training only D/G heads by rebuilding the optimizer.

import time, sys, random, os, copy
import numpy as np
import torch
from datetime import timedelta
import subprocess


# ------------------------------ GPU utilities ------------------------------

def get_nvidia_smi_output():
    nvidia_smi = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
    nvidia_smi_output = nvidia_smi.communicate()[0].decode('utf8')
    return nvidia_smi_output.split('\n')

def parse_gpu_usages(nvidia_smi_lines):
    usages = []
    for line in nvidia_smi_lines:
        str_idx = line.find('MiB / ')
        if str_idx != -1:
            usages.append(int(line[str_idx-7:str_idx]))
    return usages

def pick_lowest_usage_gpu(usages, pick_from):
    gpus_sorted = np.argsort(usages)
    for idx in gpus_sorted:
        if idx in pick_from:
            return 'cuda:' + str(idx)
    return 'cpu'

def GetLowestGPU(pick_from=[0, 1, 2, 3], verbose=True, return_usages=False, mps=False, cpu=False):

    if cpu:
        if verbose:
            print('Device set to cpu')
        return 'cpu'
    if not torch.cuda.is_available() or not pick_from:
        if mps:
            print('Device set to mps')
            return 'mps'
        if verbose:
            print('Device set to cpu')
        return 'cpu'
    nvidia_smi_lines = get_nvidia_smi_output()
    usages = parse_gpu_usages(nvidia_smi_lines)
    device = pick_lowest_usage_gpu(usages, pick_from)
    if verbose:
        print(" ======================= GPU USAGES ================")
        print('Device set to ' + device)
        print("=====================================================")
    if return_usages:
        return device, usages
    else:
        return device

def synchronize_if_needed(x):
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    elif x.device.type == "mps":
        torch.mps.synchronize()
    # no sync needed for CPU

# ------------------------------ Time helper --------------------------------

def TimeRemaining(current_iter, 
                  total_iter, 
                  start_time, 
                  previous_time=None, 
                  ops_per_iter=1.0):
    current_time = time.time()
    elapsed = current_time - start_time
    remaining = total_iter * elapsed / current_iter - elapsed
    ms_per_op = None
    if previous_time is not None:
        ms_per_op = (current_time - previous_time) / ops_per_iter
    elapsed = str(timedelta(seconds=int(elapsed)))
    remaining = str(timedelta(seconds=int(remaining)))
    return elapsed, remaining, ms_per_op

# ------------------------------ Tensor move helpers ------------------------

def _to_device_obj(obj, device):
    """Return obj moved to device if it's a Tensor/collection containing Tensors."""
    try:
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        elif isinstance(obj, (list, tuple)):
            seq_type = type(obj)
            return seq_type(_to_device_obj(x, device) for x in obj)
        elif isinstance(obj, dict):
            return {k: _to_device_obj(v, device) for k, v in obj.items()}
    except Exception:
        pass
    return obj

def _move_unregistered_tensors_in_module(module, device, _visited=None):
    """Move any Tensor attributes not registered as params/buffers to device."""
    if _visited is None:
        _visited = set()
    if id(module) in _visited:
        return
    _visited.add(id(module))

    # Collect names of registered params/buffers to avoid redundant sets
    registered = set(name for name, _ in module.named_parameters(recurse=False))
    registered.update(name for name, _ in module.named_buffers(recurse=False))

    # Traverse direct attributes
    for name, val in list(vars(module).items()):
        if name in registered:
            continue  # already handled by module.to(device)
        if isinstance(val, torch.nn.Module):
            _move_unregistered_tensors_in_module(val, device, _visited)
            continue
        new_val = _to_device_obj(val, device)
        if new_val is not val:
            try:
                setattr(module, name, new_val)
            except Exception:
                # Some attributes may be read-only properties; ignore safely
                pass

# ------------------------------ Model Wrapper -------------------------------
#print("""
# MODULE | `ModelWrapper` | simplified version – 23 May 25
#
# Info:
# - Removed per‑batch closure pattern.
# - Removed verbose level 2.
# - Removed all type annotations to declutter.
# - Removed *all* callback hooks.
#""")

import time, sys, random
import numpy as np
import torch

from datetime import timedelta
#def print_param_sample(mw, n=10):
 #   model = mw.model.surface_fitter
  #  params = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
   # print(params[:n])

class ModelWrapper:
    """Lightweight helper around a PyTorch model.

    Key differences vs the legacy wrapper:
        • Direct forward/backward loop (optimisers that require a *closure* like
          LBFGS are no longer supported).
        • Only epoch‑level logging (`verbose=1`) – otherwise silent.
        • No callback hooks.
        • No static type hints (cleaner source).
    """

    # ------------------------------------------------------------------
    # CONSTRUCTOR -------------------------------------------------------
    # ------------------------------------------------------------------

    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 regularizer=None,
                 #augmentation=None,
                 #scheduler=None,
                 save_name=None,
                 save_best_train=False,
                 save_best_val=True,
                 save_opt=False,
                 save_reg=False,
                 seed=0):

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.regularizer = regularizer
        #self.augmentation = augmentation
        #self.scheduler = scheduler
        self.save_name = save_name
        self.save_best_train = bool(save_name and save_best_train)
        self.save_best_val = bool(save_name and save_best_val)
        self.save_opt = bool(save_name and save_opt)
        self.save_reg = bool(save_name and save_reg)
        self.seed = seed


        # Logs ---------------------------------------------------------
        self.train_loss_list = []
        self.val_loss_list = []
        self.train_pde_loss_list = []
        self.val_pde_loss_list = []
        self.train_data_loss_list = []
        self.val_data_loss_list = []
        self.epoch_times = []

        self.diffusion_errors, self.growth_errors = [],[]
        self.diffusion_preds,self.growth_preds = [], []

        # Optional spatial snapshots ----------------------------------
        self.train_pde_losses_list_spatial = []
        self.train_data_losses_list_spatial = []
        self.x_train_list = []
        self.loss_count_list = []
        self.save_index = []

        self.train = False
        self.val = False

        # flags for u freezing logic
        self.u_frozen = False
        self.u_unfrozen_after_ES = False   # did we already unfreeze after ES_freeze?

        if self.seed is not None:
            self.set_seed(self.seed)

    # ------------------------------------------------------------------
    # TRAINING LOOP -----------------------------------------------------
    # ------------------------------------------------------------------

    def fit(self,
            x_tr_input,
            y_tr_input,
            *,
            batch_size=None,
            epochs=1,
            verbose=1,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            early_stopping=None,
            #best_train_loss=None,
            #best_val_loss=None,
            include_val_aug=False,
            include_val_reg=False,
            lr_dec_epoch=None,
            lr_dec_prop=1.0,
            rel_update_thresh = 0.01,
            rel_save_thresh=0.01,
            freeze_u_after_epoch=None,  # epoch to start freezing u (surface_fitter)
            ES_freeze=None,#10,             # if no val improv for this many epochs WHILE FROZEN -> unfreeze u
            print_freq=100,
            ):

        if self.seed is not None:
            self.set_seed(self.seed)

        self.early_stopping = early_stopping

        if batch_size is None:
            batch_size = len(x_tr_input)
        train_batches_per_epoch = max(1, len(x_tr_input) // batch_size)

        if validation_data is not None:
            x_val, y_val = validation_data
            val_batch_size = batch_size
            val_batches_per_epoch = max(1, len(x_val) // val_batch_size)

        self.best_train_loss = getattr(self, "best_train_loss", float("inf"))
        self.best_val_loss = getattr(self, "best_val_loss", float("inf"))
        self.last_improved = getattr(self, "last_improved", 0)
        self.max_trigger = getattr(self, "max_trigger", 0)
        self.trigger_list = getattr(self, "trigger_list", [])
        global_start_time = time.time()
        self.print_freq = print_freq
        self.avg_epoch_time = None



        for epoch in range(initial_epoch, initial_epoch + epochs):


            
            # First check if we should stop training
            # ================================================
            trigger = self.model.epochs - self.last_improved
            if trigger > self.max_trigger:
                    self.max_trigger = trigger

            if not self.u_frozen and trigger >= early_stopping:
                print("\n\nEarly stopping – no improvement.")
                self.save(f"{self.save_name}_ES")
                print(f"Saved model with early stopping at epoch {self.model.epochs}")
                break
            #=================================================


            # Manage freezing / unfreezing of u-network
            self._maybe_manage_freeze(
                epoch=epoch,
                trigger=trigger,
                freeze_u_after_epoch=freeze_u_after_epoch,
                ES_freeze=ES_freeze
            )


            self.train, self.val = True, False
            epoch_start_time = time.time()
            self.model.train()
            

            #self.set_seed(self.model.epochs)
            torch.manual_seed(self.model.epochs) 
            # Shuffle once per epoch ----------------------------------
            perm = torch.randperm(len(x_tr_input))
            x_tr = x_tr_input[perm].data
            y_tr = y_tr_input[perm].data

            epoch_train_losses = []
            epoch_train_pde = []
            epoch_train_data = []



            # ---------------- batch loop ----------------------------
            for batch_idx in range(train_batches_per_epoch):
                if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                    break

                start = batch_idx * batch_size
                stop = (batch_idx + 1) * batch_size if batch_idx + 1 < train_batches_per_epoch else len(x_tr)

                x_true = x_tr[start:stop]
                y_true = y_tr[start:stop]

                #if self.augmentation is not None:
                 #   x_true, y_true = self.augmentation(x_true, y_true)

                x_true.requires_grad_(True)
                self.optimizer.zero_grad(set_to_none=True)

                y_pred = self.model(x_true)

                #print("y_pred", y_pred[:3])
                

                losses = self.loss(y_pred, y_true)  # expected tuple
                task_loss, data_loss, pde_loss = losses[:3]
                #data_loss_full, pde_loss_full = losses[3:5]

                reg_loss = self.regularizer(self.model, x_true, y_true, y_pred) if self.regularizer is not None else 0.0
                total_loss = task_loss + reg_loss
                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                
                #if self.scheduler is not None:
                 #   self.scheduler.step()

                epoch_train_losses.append(total_loss.detach())
                epoch_train_pde.append(pde_loss.detach())
                epoch_train_data.append(data_loss.detach())

                if hasattr(self.model, 'epochs') and self.model.epochs % 100 == 0:
                    self.save_index.append(self.model.loss_count - 1)
                #    self.train_data_losses_list_spatial.append(data_loss_full.detach())
                 #   self.train_pde_losses_list_spatial.append(pde_loss_full.detach())
                  #  self.x_train_list.append(x_true.detach())

            # ---------------- end batch loop -------------------------
            train_loss_epoch = torch.mean(torch.stack(epoch_train_losses)).item()
            train_pde_epoch = torch.mean(torch.stack(epoch_train_pde)).item()
            train_data_epoch = torch.mean(torch.stack(epoch_train_data)).item()

            self.train_loss_list.append(train_loss_epoch)
            self.train_pde_loss_list.append(train_pde_epoch)
            self.train_data_loss_list.append(train_data_epoch)

            # Best‑train tracking
            #if train_loss_epoch < best_train_loss * (1 - rel_save_thresh):
             #   best_train_loss = train_loss_epoch
              #  self.last_improved = epoch
               # if self.save_best_train and self.save_name:
                #    self.save(f"{self.save_name}_best_train")

            # Validation step
            if validation_data is not None and (epoch % validation_freq == 0):
                self._validate(x_val, y_val,
                               val_batches_per_epoch, val_batch_size,
                               validation_steps,
                               include_val_aug, include_val_reg,
                               self.best_val_loss, rel_save_thresh)
                
                if self.val_loss_list[-1] <self.best_val_loss * (1 - rel_update_thresh):
                    self.best_val_loss = self.val_loss_list[-1]
                    self.last_improved = self.model.epochs

                    self.best_diffusion_pred = self.model.D_scale * self.model.diffusion(self.model.u_vals_torch).flatten()
                    self.best_diffusion_error = torch.mean((self.model.D_true_torch - self.best_diffusion_pred)**2).item()

                    if self.model.growth:
                        self.best_growth_pred = self.model.G_scale * self.model.growth(self.model.u_vals_torch).flatten()
                        self.best_growth_error = torch.mean((self.model.G_true_torch - self.best_growth_pred)**2).item()
                
            if validation_data is None:
                if self.train_loss_list[-1] <self.best_val_loss * (1 - rel_update_thresh):
                    self.best_val_loss = self.train_loss_list[-1]
                    self.last_improved = self.model.epochs

                    self.best_diffusion_pred = self.model.D_scale * self.model.diffusion(self.model.u_vals_torch).flatten()
                    self.best_diffusion_error = torch.mean((self.model.D_true_torch - self.best_diffusion_pred)**2).item()

                    if self.model.growth:
                        self.best_growth_pred = self.model.G_scale * self.model.growth(self.model.u_vals_torch).flatten()
                        self.best_growth_error = torch.mean((self.model.G_true_torch - self.best_growth_pred)**2).item()

                    
            # Save best‑train snapshot
                    if self.train_loss_list[-1] < self.best_val_loss * (1 - rel_save_thresh):
                        if self.save_best_val and self.save_name:
                            self.save(f"{self.save_name}_best_val")
                        self.load_best_val(device=x_tr.device)
                    
                    

            # Learning‑rate schedule & early stop
            #if lr_dec_epoch and (epoch + 1) % lr_dec_epoch == 0:
             #   for pg in self.optimizer.param_groups:
              #      pg['lr'] *= lr_dec_prop


            if self.model.epochs%100 == 0:

                diff_pred = self.model.D_scale * self.model.diffusion(self.model.u_vals_torch).flatten()
                diffusion_error =  (self.model.D_true_torch - diff_pred)**2
                self.diffusion_errors.append(torch.mean(diffusion_error).item())
                #print("================================",torch.mean(diffusion_error) )
                self.diffusion_preds.append(diff_pred)
                
                if self.model.growth:
                    growth_pred = self.model.G_scale * self.model.growth(self.model.u_vals_torch).flatten()
                    growth_error = (self.model.G_true_torch - growth_pred)**2
                    self.growth_preds.append(growth_pred)
                    self.growth_errors.append(torch.mean(growth_error).item())

            synchronize_if_needed(x_tr)

            # Epoch‑level progress message
            if verbose == 1  and epoch%self.print_freq==0:

                elapsed, remaining, _ = TimeRemaining(
                    current_iter=self.model.epochs + 1,
                    total_iter=initial_epoch + epochs,
                    start_time=global_start_time,
                    previous_time=epoch_start_time,
                    ops_per_iter=batch_size)
                msg = (f"\rEpoch {self.model.epochs + 1}/{initial_epoch + epochs} | "
                       f"Train loss: {train_loss_epoch:1.4e}")
                if validation_data is not None:
                    msg += f" | Val loss: {self.val_loss_list[-1]:1.4e}"
                msg += f" | Remaining: {remaining}        "
                msg += f' | Trigger = {trigger}'
                msg += f' | Elapsed = {epoch_start_time-global_start_time:.1f} s'
                msg += f' | Max trigger = {self.max_trigger}'
                msg += f' | D error ={self.diffusion_errors[-1]:.3e}'
                if self.u_frozen:
                    msg += ' | u_frozen=1'
                else:
                    msg += ' | u_frozen=0'

                if self.model.growth:
                    msg += f' | G error ={self.growth_errors[-1]:.3e}'

                # Clear line and print message
                print(msg, end='\r', flush=True)

            
            # monitor epoch runtime
            if epoch>0:
                self.epoch_times.append(time.time()-epoch_start_time)
            # ++++++++++++++++++++++++++++++ final print readout for verbose 1 ++++++++++++++++++++++++
        
            # Epoch leve


            if hasattr(self.model, 'epochs'):
                self.model.epochs += 1

        if early_stopping is None or trigger < early_stopping:
            print("\nNumber of epochs to train finished rather than early stopping.")
            self.save(f"{self.save_name}_expired")
            print(f"Saved model at total trained epochs {self.model.epochs}")

        if verbose == 1:
            print("\nTraining finished.")
            print("\nTotal epochs trained =", self.model.epochs)
            print(f"\nBest D error ={self.best_diffusion_error:.3e}")
            if self.model.growth:
                print(f"\nBest G error ={self.best_growth_error:.3e}")
            #print(f"\nBest train loss = {self.best_train_loss:.3e}")
            print(f"\nBest val loss = {self.best_val_loss:.3e}")


        #if self.epoch_times:
         #   self.avg_epoch_time = np.mean(self.epoch_times)/self.print_freq


    # ------------------------------------------------------------------
    # VALIDATION --------------------------------------------------------
    # ------------------------------------------------------------------

    def _validate(self,
                  x_val, y_val,
                  val_batches_per_epoch, val_batch_size,
                  validation_steps,
                  include_val_aug, include_val_reg,
                  best_val_loss, rel_save_thresh):
        #self.train, self.val = False, True
        self.model.eval()

        val_loss_acc = 0.0
        val_pde_acc = 0.0
        val_data_acc = 0.0
        val_reg_acc = 0.0

        #with torch.no_grad():
        for idx in range(val_batches_per_epoch):
            if validation_steps is not None and idx >= validation_steps:
                break

            start = idx * val_batch_size
            stop = (idx + 1) * val_batch_size if idx + 1 < val_batches_per_epoch else len(x_val)
            x_true = x_val[start:stop].clone()
            y_true = y_val[start:stop].clone()

            if include_val_aug and self.augmentation is not None:
                x_true, y_true = self.augmentation(x_true, y_true)

            x_true.requires_grad_(True)
            y_pred = self.model(x_true)

            losses = self.loss(y_pred, y_true)
            val_loss_acc += losses[0]
            val_data_acc += losses[1]
            val_pde_acc += losses[2]

            if include_val_reg and self.regularizer is not None:
                val_reg_acc += self.regularizer(self.model, x_true, y_true, y_pred)

        batches = idx + 1
        val_total = (val_loss_acc + val_reg_acc) / batches
        val_pde = (val_pde_acc + val_reg_acc) / batches
        val_data = (val_data_acc + val_reg_acc) / batches

  
        self.val_loss_list.append(val_total.item())
        self.val_pde_loss_list.append(val_pde.item())
        self.val_data_loss_list.append(val_data.item())

        synchronize_if_needed(x_true)

        # Save best‑val snapshot
        if val_total < best_val_loss * (1 - rel_save_thresh):
            if self.save_best_val and self.save_name:
                self.save(f"{self.save_name}_best_val")
            self.load_best_val(device=x_val.device)

    # ------------------------------------------------------------------
    # FREEZE / OPTIMIZER UTILITIES -------------------------------------
    # ------------------------------------------------------------------

    def _rebuild_optimizer_for_dg(self):
        """
        Rebuild optimizer so it only updates D/G head parameters.
        Assumes model provides .dg_parameters() -> iterable[Tensor].
        """
        if hasattr(self.model, "dg_parameters"):
            params = list(self.model.dg_parameters())
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]

        if len(params) == 0:
            raise RuntimeError("No trainable parameters found for D/G heads.")

        old_opt = self.optimizer
        opt_cls = type(old_opt)

        base_group = old_opt.param_groups[0].copy()
        lr = base_group.pop("lr", None)
        base_group.pop("params", None)

        if lr is None:
            self.optimizer = opt_cls(params, **base_group)
        else:
            self.optimizer = opt_cls(params, lr=lr, **base_group)

    def _rebuild_optimizer_for_full(self):
        """
        Rebuild optimizer over all trainable parameters (requires_grad=True).
        """
        params = [p for p in self.model.parameters() if p.requires_grad]

        if len(params) == 0:
            raise RuntimeError("No trainable parameters found for full model.")

        old_opt = self.optimizer
        opt_cls = type(old_opt)

        base_group = old_opt.param_groups[0].copy()
        lr = base_group.pop("lr", None)
        base_group.pop("params", None)

        if lr is None:
            self.optimizer = opt_cls(params, **base_group)
        else:
            self.optimizer = opt_cls(params, lr=lr, **base_group)

    def _maybe_manage_freeze(self, epoch, trigger, freeze_u_after_epoch, ES_freeze):
        """
        Handles:
          1) Freezing u-network at freeze_u_after_epoch and switching optimizer to D/G.
          2) Unfreezing u if ES_freeze epochs pass with no val improvement while frozen.

        Guarantees at most:
          - 1 freeze event
          - 1 unfreeze event
        """
        current_epoch = getattr(self.model, "epochs", epoch)

        # ---- Step 1: freeze u at target epoch (only if never unfrozen after ES) ----
        if (freeze_u_after_epoch is not None and
            (not self.u_frozen) and           # not currently frozen
            (not self.u_unfrozen_after_ES) and # haven't already done the unfreeze
            current_epoch >= freeze_u_after_epoch):

            # Freeze surface network if method exists
            if hasattr(self.model, "freeze_surface"):
                self.model.freeze_surface(True)
            else:
                # Fallback: freeze parameters of surface_fitter if present
                if hasattr(self.model, "surface_fitter"):
                    for p in self.model.surface_fitter.parameters():
                        p.requires_grad = False

            self._rebuild_optimizer_for_dg()
            self.u_frozen = True
            print(f"\n[ModelWrapper] Froze u/surface_fitter at epoch {current_epoch} "
                  f"and rebuilt optimizer for D/G only.")

        # ---- Step 2: unfreeze u if ES_freeze reached while frozen (only once) ----
        if (self.u_frozen and
            ES_freeze is not None and
            (not self.u_unfrozen_after_ES) and
            trigger >= ES_freeze):

            # Unfreeze surface network
            if hasattr(self.model, "freeze_surface"):
                self.model.freeze_surface(False)
            else:
                if hasattr(self.model, "surface_fitter"):
                    for p in self.model.surface_fitter.parameters():
                        p.requires_grad = True

            # Rebuild optimizer over full model
            self._rebuild_optimizer_for_full()
            self.u_frozen = False
            self.u_unfrozen_after_ES = True
            print(f"\n[ModelWrapper] Unfroze u/surface_fitter at epoch {current_epoch} "
                  f"after {trigger} epochs without improvement while frozen.")
            self.last_improved = self.model.epochs 
            self.max_trigger = 0  # reset max_trigger after unfreezing
            self.frozen_switch_epoch = current_epoch
           

    # ------------------------------------------------------------------
    # UTILITIES ---------------------------------------------------------
    # ------------------------------------------------------------------


     #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    def set_seed(self, seed):
        """
        Sets the random seed for reproducibility across Python, NumPy, and PyTorch.

        Args:
            seed (int): The seed value to set.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def predict(self, inputs):
        
        """
        Runs the model in evaluation mode on the provided inputs.

        Args:
            inputs (torch.Tensor): Input data to pass through the model.

        Returns:
            torch.Tensor: Model predictions for the input data.
        """
        
        # run model in eval mode (for batchnorm, dropout, etc.)
        self.model.eval()
        
        return self.model(inputs)
    
    def save(self, save_name):
        """
        Saves the model, optimizer, and regularizer weights.

        Args:
            save_name (str): The base name for saving the weights.
        """
        # Save model weights
        model_name = "{}_model".format(save_name)
        
        torch.save(self.model.state_dict(), model_name)
        
        # Save additional metadata
        torch.save({'epochs': self.model.epochs}, 
                   "{}_epochs".format(save_name))

        # Save optimizer weights if required
        if self.save_opt and self.optimizer:
            torch.save(self.optimizer.state_dict(), 
                       "{}_opt".format(save_name))

        # Save regularizer weights if required
        if self.save_reg and self.regularizer:
            torch.save(self.regularizer.state_dict(),
                       "{}_reg".format(save_name))

    
    def load(self, 
             model_weights, 
             opt_weights=None, 
             reg_weights=None, 
             device=None):
        """
        Loads the model, optimizer, and regularizer weights.

        Args:
            model_weights (str): Path to the model weights file.
            opt_weights (str, optional): Path to the optimizer weights file. Defaults to None.
            reg_weights (str, optional): Path to the regularizer weights file. Defaults to None.
            device (str, optional): Device to map the loaded weights onto. Defaults to None.
        """
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=False))
        self.model.eval()

        # Load optimizer weights if provided
        if opt_weights:
            self.optimizer.load_state_dict(torch.load(opt_weights, map_location=device, weights_only=False))

        # Load regularizer weights if provided
        if reg_weights:
            self.regularizer.load_state_dict(torch.load(reg_weights, map_location=device, weights_only=False))

    

    def load_best_train(self, device=None, intro = ''):
        
        """
        Loads the model weights that achieved the best training error.

        Args:
            device (str, optional): Device to map the loaded weights onto. Defaults to None.
        """
        
        self._load_best_weights(suffix='best_train', device=device, intro=intro)
    
    def load_best_val(self, device=None, intro = ''):
        """
        Loads the model weights that achieved the best validation error.

        Args:
            device (str, optional): Device to map the loaded weights onto. Defaults to None.
    """
        self._load_best_weights(suffix='best_val', device=device, intro=intro)

    def load_ES(self, device=None, intro = ''):
        """
        Loads the model weights that achieved the best validation error.

        Args:
            device (str, optional): Device to map the loaded weights onto. Defaults to None.
    """
        self._load_best_weights(suffix='ES', device=device, intro=intro)

    def load_expired(self, device=None, intro = ''):
        """
        Loads the model weights that achieved the best validation error.

        Args:
            device (str, optional): Device to map the loaded weights onto. Defaults to None.
    """
        self._load_best_weights(suffix='expired', device=device, intro=intro)

            
    def _load_best_weights(self, suffix, device, intro = ''):

        """
        Helper function to load model, optimizer, and regularizer weights for best training or validation error.

        Args:
            suffix (str): Suffix indicating whether to load 'best_train' or 'best_val' weights.
            device (str, optional): Device to map the loaded weights onto. Defaults to None.
        """
        # Load model weights
        model_name = intro + "{}_{}_model".format(self.save_name,suffix)
        self.model.load_state_dict(torch.load(model_name, map_location=device, weights_only=False))
        self.model.eval()

        # Load optimizer weights if applicable
        if self.save_opt and self.optimizer:
            opt_name = "{}_{}_opt".format(self.save_name,suffix)
            self.optimizer.load_state_dict(torch.load(opt_name, map_location=device, weights_only=False))

        # Load regularizer weights if applicable
        if self.save_reg and self.regularizer:
            reg_name = "{}_{}_reg".format(self.save_name,suffix)
            self.regularizer.load_state_dict(torch.load(reg_name, map_location=device, weights_only=False))


def TimeRemaining(current_iter, 
                  total_iter, 
                  start_time, 
                  previous_time=None, 
                  ops_per_iter=1.0):
    
    '''
    Computes time remaining in a loop.
    
    Args:
        current_iter:  integer for current iteration number
        total_iter:    integer for total number of iterations
        start_time:    float initial time
        previous_time: float time of previous iteration
        ops_per_iter:  integer number of operations per iteration
        
    Returns:
        elapsed:   string of elapsed time
        remaining: string of remaining time
        ms_per_op: optional string of milliseconds per operation
    '''
    
    # compute elapsed and remaining time
    current_time = time.time()
    elapsed = current_time - start_time
    remaining = total_iter * elapsed / current_iter - elapsed
    
    # compute optional time between operations
    ms_per_op = None
    if previous_time is not None:
        ms_per_op = (current_time - previous_time) / ops_per_iter
    
    # convert seconds to datetime
    elapsed = str(timedelta(seconds=int(elapsed)))
    remaining = str(timedelta(seconds=int(remaining)))
    
    # convert seconds to milliseconds
    #if ms_per_op is not None:
     #   ms_per_op = '{0}'.format(int(np.round(ms_per_op * 1000)))
        
    return elapsed, remaining, ms_per_op

# Version 19 Dec 24

# Taken from [1].

# References
# ----------
# [1] Lagergren JH, Nardini JT, Baker RE, Simpson MJ, Flores KB (2020) Biologically-
#      informed neural networks guide mechanistic modeling from sparse experimental 
#      data. PLoS Comput Biol 16(12): e1008462. # https://doi.org/10.1371/journal.pcbi.1008462

import subprocess
import torch
import numpy as np

def get_nvidia_smi_output():
    """
    Executes the `nvidia-smi` command and returns its output as a list of strings.
    Each line of the command output is a separate list element.
    
    Returns:
        nvidia_smi_lines: List of strings representing `nvidia-smi` output.
    """
    nvidia_smi = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
    nvidia_smi_output = nvidia_smi.communicate()[0].decode('utf8')
    return nvidia_smi_output.split('\n')

def parse_gpu_usages(nvidia_smi_lines):
    """
    Parses GPU memory usages from the `nvidia-smi` output.
    
    Args:
        nvidia_smi_lines: List of strings from `nvidia-smi` output.
    
    Returns:
        usages: List of integer memory usages for all GPUs.
    """
    usages = []
    for line in nvidia_smi_lines:
        str_idx = line.find('MiB / ')
        if str_idx != -1:
            usages.append(int(line[str_idx-7:str_idx]))
    return usages

def pick_lowest_usage_gpu(usages, pick_from):
    """
    Selects the GPU with the lowest memory usage from the given set of GPUs.
    
    Args:
        usages: List of integer memory usages for all GPUs.
        pick_from: List of GPU indices to choose from.
    
    Returns:
        device: String representing the chosen GPU device (e.g., 'cuda:0').
    """
    gpus_sorted = np.argsort(usages)
    for idx in gpus_sorted:
        if idx in pick_from:
            return 'cuda:' + str(idx)
    return 'cpu'

def GetLowestGPU(pick_from=[0, 1, 2, 3], verbose=True, return_usages=False, mps=False):
    """
    Determines the GPU with the lowest memory usage or falls back to CPU.
    
    Args:
        pick_from: List of GPUs to choose from.
        verbose: Whether to print the chosen device.
        return_usages: Whether to return memory usages for all GPUs.
    
    Returns:
        device: Chosen device string (e.g., 'cuda:0' or 'cpu').
        usages: Optional list of memory usages for all GPUs.
    """
    if not torch.cuda.is_available() or not pick_from:
        if mps:
            print('Device set to mps')
            return 'mps'
	
        if verbose:
            print('Device set to cpu')
        return 'cpu'

    nvidia_smi_lines = get_nvidia_smi_output()
    usages = parse_gpu_usages(nvidia_smi_lines)
    device = pick_lowest_usage_gpu(usages, pick_from)

    if verbose:
        print('Device set to ' + device)
    if return_usages:
        return device, usages
    else:
        return device
    

def synchronize_if_needed(x):
    if x.device.type == "cuda":
        torch.cuda.synchronize()
    elif x.device.type == "mps":
        torch.mps.synchronize()
    # no sync needed for CPU

# usage
