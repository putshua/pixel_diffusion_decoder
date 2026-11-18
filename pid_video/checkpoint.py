"""Strict portable FSDP/DCP model + flattened Adam state, matching the training run."""
from torch.distributed.checkpoint.state_dict import (
 StateDictOptions,get_model_state_dict,get_optimizer_state_dict,
 set_model_state_dict,set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
class ModelState(Stateful):
 def __init__(self,model):self.model=model
 def state_dict(self):return get_model_state_dict(self.model)
 def load_state_dict(self,state_dict):set_model_state_dict(self.model,state_dict,options=StateDictOptions(strict=True))
class OptimizerState(Stateful):
 def __init__(self,model,optimizer):self.model,self.optimizer=model,optimizer
 def state_dict(self):return get_optimizer_state_dict(self.model,self.optimizer,options=StateDictOptions(flatten_optimizer_state_dict=True))
 def load_state_dict(self,state_dict):set_optimizer_state_dict(self.model,self.optimizer,state_dict,options=StateDictOptions(flatten_optimizer_state_dict=True))
