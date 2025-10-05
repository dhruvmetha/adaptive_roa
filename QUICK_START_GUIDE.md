# Quick Start Guide: Universal Flow Matching Framework

## 🚀 Installation & Setup

### 1. **Install Theseus Dependencies**
```bash
# Activate your environment
conda activate /common/users/dm1487/envs/arcmg

# Install system dependencies via conda
conda install -c conda-forge suitesparse

# Install Theseus
pip install theseus-ai

# Verify installation
python -c "import theseus as th; print('✅ Theseus ready!')"
```

### 2. **Install Framework**
```bash
cd /common/home/dm1487/robotics_research/tripods/olympics-classifier
pip install -e .
```

## 🎯 Training Examples

### **Train Pendulum (S¹ × ℝ)**
```bash
python train_universal.py system.name=pendulum
```

### **Train CartPole (ℝ² × S¹ × ℝ)**  
```bash
python train_universal.py system.name=cartpole
```

### **Custom Configuration**
```bash
python train_universal.py \
    system.name=pendulum \
    system.params.attractor_radius=0.05 \
    model.hidden_dims=[32,64,128] \
    num_integration_steps=50 \
    trainer.max_epochs=200
```

## 🔬 Demo & Testing

### **Run Framework Demo**
```bash
python demo_universal_pendulum.py
```

### **Test Theseus Consistency**
```bash
python -c "
from src.systems.pendulum_universal import PendulumSystem
from src.flow_matching.universal import UniversalFlowMatcher
import torch

system = PendulumSystem()
print(f'System: {system}')
print(f'Dimensions: {system.state_dim} → {system.embedding_dim} → {system.tangent_dim}')
print('✅ Framework ready!')
"
```

## 📊 Inference Examples

### **Simple Prediction**
```python
from src.systems import PendulumSystem
from src.flow_matching.universal import UniversalFlowMatchingInference

# Load system and model
system = PendulumSystem()
inferencer = UniversalFlowMatchingInference("checkpoint.ckpt", system)

# Predict single endpoint
endpoint = inferencer.predict_single_from_components(
    angle=1.5,              # θ = 1.5 rad
    angular_velocity=0.0    # θ̇ = 0.0 rad/s
)
print(f"Predicted endpoint: θ={endpoint[0]:.3f}, θ̇={endpoint[1]:.3f}")
```

### **Full Trajectory**
```python
import torch

start_state = torch.tensor([1.5, 0.0])  # (θ, θ̇)
endpoint, path = inferencer.predict_endpoint(start_state, return_path=True)

print(f"Integration path shape: {path.shape}")  # [num_steps+1, 2]
print(f"Final endpoint: {endpoint}")
```

### **Batch Prediction**
```python
import numpy as np

# Multiple start states
start_states = [
    [0.0, 1.0],      # Bottom with velocity
    [np.pi/2, 0.0],  # Quarter turn at rest  
    [np.pi, 0.5]     # Top with velocity
]

endpoints = inferencer.batch_predict(start_states)
print(f"Batch predictions shape: {endpoints.shape}")  # [3, 2]
```

## 🛠️ Adding New Systems

### **Step 1: Define System**
```python
# src/systems/my_system.py
from .base import DynamicalSystem, ManifoldComponent

class MySystem(DynamicalSystem):
    def define_manifold_structure(self):
        return [
            ManifoldComponent("Real", 2, "position"),     # (x, y) ∈ ℝ²
            ManifoldComponent("SO2", 1, "orientation"),   # θ ∈ S¹  
            ManifoldComponent("Real", 3, "velocities")    # (vx, vy, ω) ∈ ℝ³
        ]
    
    def define_state_bounds(self):
        return {
            "position": (-10.0, 10.0),
            "orientation": (-np.pi, np.pi), 
            "velocities": (-5.0, 5.0)
        }
```

### **Step 2: Register System**
```python
# train_universal.py - add to SYSTEM_REGISTRY
SYSTEM_REGISTRY = {
    "pendulum": PendulumSystem,
    "cartpole": CartPoleSystem, 
    "my_system": MySystem,  # ← Add your system
}
```

### **Step 3: Train**
```bash
python train_universal.py system.name=my_system
```

**The framework automatically handles:**
- ✅ Model architecture sizing
- ✅ Theseus integration for SO(2) components  
- ✅ Target generation using proper log maps
- ✅ State embedding/extraction

## 📈 Monitoring Training

### **TensorBoard**
```bash
tensorboard --logdir outputs/
```

### **Check System Info**
```python
from src.flow_matching.universal import UniversalFlowMatchingConfig
from src.systems import PendulumSystem

system = PendulumSystem()
config = UniversalFlowMatchingConfig.for_system(system)

print("System Information:")
for key, value in config.get_system_info().items():
    print(f"  {key}: {value}")
```

## 🔧 Configuration Files

### **System-Specific Configs**
```yaml
# configs/train_universal_flow_matching.yaml
system:
  name: "pendulum"
  params:
    attractor_radius: 0.1

model:
  hidden_dims: [64, 128, 256]
  time_emb_dim: 128

num_integration_steps: 100
```

### **Data Configuration**
```yaml
data:
  _target_: src.data.circular_endpoint_data.CircularEndpointDataModule
  data_file: "data/pendulum_endpoints.txt"
  batch_size: 128
  num_workers: 4
```

## 🐛 Troubleshooting

### **Theseus Import Error**
```python
# Check if Theseus is working
try:
    import theseus as th
    print("✅ Theseus available")
except ImportError:
    print("❌ Install Theseus: pip install theseus-ai")
```

### **Dimension Mismatch**
```python
# Check system dimensions
system = YourSystem()
print(f"State: {system.state_dim}D")
print(f"Embedding: {system.embedding_dim}D") 
print(f"Tangent: {system.tangent_dim}D")
print(f"Model I/O: {system.embedding_dim * 2}D → {system.tangent_dim}D")
```

### **Integration Issues**
```bash
# Test integration step
python -c "
from src.manifold_integration import TheseusIntegrator
from src.systems import PendulumSystem
import torch

system = PendulumSystem()
integrator = TheseusIntegrator(system)

# Test single integration step
state = torch.tensor([[0.0, 1.0]])
velocity = torch.tensor([[1.0, -0.5]])
next_state = integrator.integrate_step(state, velocity, 0.01)
print(f'Integration test: {state} → {next_state}')
"
```

## 🎯 Next Steps

1. **Train your first model**: Start with pendulum system
2. **Experiment with architectures**: Try different hidden dimensions  
3. **Add your own system**: Follow the system definition pattern
4. **Scale to complex systems**: Ready for SE(3) × SO(3)ⁿ humanoids
5. **Contribute**: Framework is designed for research collaboration

## 📚 Key Files to Understand

- **`src/systems/base.py`**: Core system abstraction
- **`src/flow_matching/universal/flow_matcher.py`**: Training logic with Theseus
- **`src/manifold_integration/theseus_integrator.py`**: Geometric integration  
- **`train_universal.py`**: Universal training script
- **`UNIVERSAL_FRAMEWORK_DOCUMENTATION.md`**: Complete technical details

**Ready to explore the future of geometric flow matching! 🚀**