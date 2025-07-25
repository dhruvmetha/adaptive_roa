import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchmetrics import MeanMetric
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

class FlowMatching(pl.LightningModule):
    def __init__(self, model, optimizer, scheduler):
        super().__init__()
        
        # Initialize the neural network (UNet1D) - already instantiated by Hydra
        self.model = model
        
        # Initialize flow matcher
        self.flow_matcher = ConditionalFlowMatcher(sigma=0.0)
        
        # Optimizer and scheduler
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        
        # Metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x, t, condition):
        """
        Forward pass of the model
        x: current state in flow [batch_size, output_dim]
        t: time [batch_size]
        condition: start state [batch_size, input_dim]
        """
        return self.model(x, t, condition)
    
    def training_step(self, batch, batch_idx):
        # Get start and end states
        start_states = batch["start_state"]  # [batch_size, 2]
        end_states = batch["end_state"]      # [batch_size, 2]
        
        # Sample random times
        batch_size = start_states.shape[0]
        t = torch.rand(batch_size, device=self.device)
        
        # Get conditional flow matching loss
        # This samples x_t and computes the target velocity
        t_sampled, x_t, ut = self.flow_matcher.sample_location_and_conditional_flow(
            x0=start_states, x1=end_states, t=t
        )
        
        # Predict velocity using our model
        pred_ut = self.model(x_t, t, condition=start_states)
        
        # Compute MSE loss between predicted and target velocity
        loss = nn.functional.mse_loss(pred_ut, ut)
        
        # Log metrics
        self.train_loss.update(loss)
        self.log("train_loss", self.train_loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Get start and end states
        start_states = batch["start_state"]
        end_states = batch["end_state"]
        
        # Sample random times
        batch_size = start_states.shape[0]
        t = torch.rand(batch_size, device=self.device)
        
        # Get conditional flow matching loss
        t_sampled, x_t, ut = self.flow_matcher.sample_location_and_conditional_flow(
            x0=start_states, x1=end_states, t=t
        )
        
        # Predict velocity
        pred_ut = self.model(x_t, t, condition=start_states)
        
        # Compute loss
        loss = nn.functional.mse_loss(pred_ut, ut)
        
        # Log metrics
        self.val_loss.update(loss)
        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test by sampling from the learned flow"""
        start_states = batch["start_state"]
        end_states = batch["end_state"]
        
        # Generate samples using the learned flow
        generated_ends = self.sample(start_states, num_steps=100)
        
        # Compute MSE between generated and true endpoints
        mse_loss = nn.functional.mse_loss(generated_ends, end_states)
        
        self.log("test_mse", mse_loss, on_epoch=True, prog_bar=True)
        
        return {"test_mse": mse_loss, "generated": generated_ends, "true": end_states}
    
    @torch.no_grad()
    def sample(self, start_states, num_steps=100):
        """
        Sample from the learned flow
        start_states: [batch_size, input_dim] - conditioning states
        """
        batch_size = start_states.shape[0]
        device = start_states.device
        
        # Start from the actual start states (not random noise)
        x = start_states.clone()
        
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.ones(batch_size, device=device) * i * dt
            
            # Predict velocity
            with torch.no_grad():
                velocity = self.model(x, t, condition=start_states)
            
            # Euler step
            x = x + velocity * dt
        
        return x
    
    def configure_optimizers(self):
        optimizer = self.optimizer_partial(params=self.parameters())
        scheduler = self.scheduler_partial(optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train_loss",
            },
        }