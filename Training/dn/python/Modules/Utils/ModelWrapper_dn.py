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


class ModelWrapper_dn:
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
                 augmentation=None,
                 scheduler=None,
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
        self.augmentation = augmentation
        self.scheduler = scheduler
        self.save_name = save_name
        self.save_best_train = bool(save_name and save_best_train)
        self.save_best_val = bool(save_name and save_best_val)
        self.save_opt = bool(save_name and save_opt)
        self.save_reg = bool(save_name and save_reg)
        self.seed = seed
        self.print_freq = 1000
        self.avg_epoch_time = None


        # Logs ---------------------------------------------------------
        self.train_loss_list = []
        self.val_loss_list = []
        self.epoch_times = []
        self.u_errors = []


        # Optional spatial snapshots ----------------------------------
        self.x_train_list = []
        self.loss_count_list = []
        self.save_index = []
        self.train_data_losses_list_spatial = []
        

        self.train = False
        self.val = False

        if self.seed is not None:
            self.set_seed(self.seed)

    # ------------------------------------------------------------------
    # TRAINING LOOP -----------------------------------------------------
    # ------------------------------------------------------------------

    def fit(self,
            x_tr,
            y_tr,
            *,
            batch_size=None,
            epochs=1,
            verbose=1,
            validation_data=None,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_freq=1,
            early_stopping=5000,
            manual_termination = None,
            #self.best_train_loss=None,
            #self.best_val_loss=None,
            include_val_aug=False,
            include_val_reg=False,
            lr_dec_epoch=None,
            lr_dec_prop=1.0,
            rel_update_thresh = 0.01,
            rel_save_thresh=0.01):
        

        start_time = time.time()

        if self.seed is not None:
            self.set_seed(self.seed)

        if batch_size is None:
            batch_size = len(x_tr)
        train_batches_per_epoch = max(1, len(x_tr) // batch_size)

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
        self.train, self.val = True, False

        for epoch in range(initial_epoch, initial_epoch + epochs):
            
            epoch_start_time = time.time()
            self.model.train()

            # Shuffle once per epoch ----------------------------------

            perm = torch.randperm(len(x_tr))
            x_tr = x_tr[perm].data
            y_tr = y_tr[perm].data

            epoch_train_losses = []

            # ---------------- batch loop ----------------------------
            for batch_idx in range(train_batches_per_epoch):
                if steps_per_epoch is not None and batch_idx >= steps_per_epoch:
                    break

                start = batch_idx * batch_size
                stop = (batch_idx + 1) * batch_size if batch_idx + 1 < train_batches_per_epoch else len(x_tr)

                x_true = x_tr[start:stop]
                y_true = y_tr[start:stop]

                if self.augmentation is not None:
                    x_true, y_true = self.augmentation(x_true, y_true)

                x_true.requires_grad_(True)
                self.optimizer.zero_grad(set_to_none=True)

                y_pred = self.model(x_true)

                losses = self.loss(y_pred, y_true)  # expected tuple
                data_loss, data_loss_full = losses[:2]

                reg_loss = self.regularizer(self.model, x_true, y_true, y_pred) if self.regularizer is not None else 0.0
                total_loss = data_loss + reg_loss
                total_loss.backward(retain_graph=True)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                epoch_train_losses.append(total_loss.detach())

                if hasattr(self.model, 'epochs') and self.model.epochs % 100 == 0:
                    self.save_index.append(self.model.loss_count - 1)
                    #self.train_data_losses_list_spatial.append(data_loss_full.detach())
                    #self.x_train_list.append(x_true.detach())

            # ---------------- end batch loop -------------------------
            train_loss_epoch = torch.mean(torch.stack(epoch_train_losses)).item()

            self.train_loss_list.append(train_loss_epoch)

            # Best‑train tracking
            #if train_loss_epoch < self.best_train_loss * (1 - rel_save_thresh):
             #   self.best_train_loss = train_loss_epoch
              #  self.last_improved = epoch
               # if self.save_best_train and self.save_name:
                #    self.save(f"{self.save_name}_best_train")
            
            u_pred = self.model.u_scale * self.model.surface_fitter(self.model.torch_meshgrid).flatten()
            u_error = (u_pred - self.u_clean_torch_flat)**2
            self.u_errors.append(torch.mean(u_error).item())


            # Validation step
            if validation_data is not None and (epoch % validation_freq == 0):
                self._validate(x_val, y_val,
                               val_batches_per_epoch, val_batch_size,
                               validation_steps,
                               include_val_aug, include_val_reg,
                               self.best_val_loss, rel_save_thresh,rel_update_thresh,u_error)
                

            if validation_data is None:
                if self.train_loss_list[-1] <self.best_val_loss * (1 - rel_save_thresh):
                    if self.save_best_val and self.save_name:
                            self.save(f"{self.save_name}_best_val")
                if self.train_loss_list[-1] <self.best_val_loss * (1 - rel_update_thresh):
                    self.best_val_loss = self.train_loss_list[-1]
                    self.last_improved = self.model.epochs

                    #u_pred = self.model.u_scale * self.model.surface_fitter(self.model.torch_meshgrid).flatten()
                    #u_error = (u_pred - self.u_clean_torch_flat)**2
                    self.u_error_best = torch.mean(u_error).item()
                    self.load_best_val(device=x_tr.device)
                
                    

            # Learning‑rate schedule & early stop
            #if lr_dec_epoch and (epoch + 1) % lr_dec_epoch == 0:
             #   for pg in self.optimizer.param_groups:
              #      pg['lr'] *= lr_dec_prop]


            if manual_termination is not None and len(self.train_loss_list) >= manual_termination:
                print("\nManual termination. Epochs = ", self.model.epochs)
                self.save(f"{self.save_name}_best_terminated")
                break

            if early_stopping is not None and self.model.epochs - self.last_improved >= early_stopping:
                print("\nEarly stopping – no improvement.")
                break

            #if self.model.epochs%10 == 0:

            #   
            # Epoch‑level progress message
            if verbose == 1 and self.model.epochs%self.print_freq==0:
                
                self.trigger_list.append(self.model.epochs - self.last_improved)
 
                if self.trigger_list[-1] > self.max_trigger:
                        self.max_trigger = self.trigger_list[-1]

                elapsed, remaining, _ = TimeRemaining(
                    current_iter=epoch + 1,
                    total_iter=initial_epoch + epochs,
                    start_time=global_start_time,
                    previous_time=epoch_start_time,
                    ops_per_iter=batch_size)
                msg = (f"\rEpoch {epoch + 1}/{initial_epoch + epochs} | "
                       f"Train loss: {train_loss_epoch:1.4e}")
                if validation_data is not None:
                    msg += f" | Val loss: {self.val_loss_list[-1]:1.4e}"
                msg += f" | Remaining: {remaining}        "
                msg += f' | Trigger= {self.trigger_list[-1]}'
                msg += f' | Elapsed = {epoch_start_time-global_start_time:.1f} s'
                msg += f' | Max Trigger = {self.max_trigger}'
                msg += f' | u error ={self.u_errors[-1]:.3e}'

                # Clear line and print message
                print(msg, end='\r', flush=True)

            # monitor epoch runtime (skip first)
            if epoch>0:
                self.epoch_times.append(time.time()-epoch_start_time)


            if hasattr(self.model, 'epochs'):
                self.model.epochs += 1

        if verbose == 1:
            print("\nTraining finished.")
            print(f"\nBest u error ={self.u_error_best:.3e}")
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
                  best_val_loss, rel_save_thresh,rel_update_thresh,u_error):

        self.model.eval()

        val_loss_acc = 0.0
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

            if include_val_reg and self.regularizer is not None:
                val_reg_acc += self.regularizer(self.model, x_true, y_true, y_pred)

        batches = idx + 1
        val_total = (val_loss_acc + val_reg_acc) / batches

        self.val_loss_list.append(val_total.item())

        if val_total < self.best_val_loss * (1 - rel_save_thresh):
            if self.save_best_val and self.save_name:
                self.save(f"{self.save_name}_best_val")

        if val_total < self.best_val_loss * (1 - rel_update_thresh):
            self.best_val_loss = self.val_loss_list[-1]
            self.last_improved = self.model.epochs
            self.u_error_best = torch.mean(u_error).item()
            self.load_best_val(device=x_val.device)
        # Save best‑val snapshot

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

    def load_best_terminated(self, device=None, intro = ''):
        """
        Loads the model weights that achieved the best validation error.

        Args:
            device (str, optional): Device to map the loaded weights onto. Defaults to None.
    """
        self._load_best_weights(suffix='best_terminated', device=device, intro=intro)
            
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