import torch
from transformers import ViTModel
import pytorch_lightning as pl
import coremltools as ct
from vit.vision_transformer_graph import vit_base

# Instantiate the model and load the checkpoint
model = vit_base()
checkpoint = torch.load("vvit_hmdb51.ckpt", map_location="cpu")

# Check if the checkpoint contains 'state_dict' or is directly the model weights
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

print("Model loaded successfully with checkpoint weights.")

# Save the Lightning model as a .pt file
torch.jit.save(model.to_torchscript(), "vit_model.pt")
print("Model saved as vit_model.pt")


# # Convert .pt model directly to Core ML (.mlpackage) format
# dummy_input = torch.randn(1, 3, 224, 224)  # Define a dummy input with the correct input shape

# # Convert the model using coremltools, specifying the input shape
# mlmodel = ct.convert(
#     "vit_model.pt",
#     source="pytorch",  # Specify source as PyTorch
#     inputs=[ct.TensorType(shape=dummy_input.shape)],  # Define the input shape for Core ML
#     convert_to="mlprogram"  # Use mlprogram format for iOS 14+ compatibility
# )

# # Save the Core ML model
# mlmodel.save("vit_model.mlpackage")
# print("Model successfully converted to Core ML format and saved as vit_model.mlpackage.")



