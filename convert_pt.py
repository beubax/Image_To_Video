import torch
import pytorch_lightning as pl
import coremltools as ct
from vit.vision_transformer_graph import vit_base
torch.cuda.empty_cache()
# Instantiate the model and load the checkpoint
model = vit_base(num_classes=51)
checkpoint = torch.load("epoch=2.ckpt", map_location="cpu")

# Check if the checkpoint contains 'state_dict' or is directly the model weights
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

print("Model loaded successfully with checkpoint weights.")
# torch.save(model.state_dict(), "vit_model.pt")

model.to(torch.device("cuda"))

# Save the Lightning model as a .pt file
example_input_tensor = torch.randn(1, 3, 16, 224, 224)
example_input_tensor = example_input_tensor.to(torch.device("cuda"))
output = model(example_input_tensor)
print(output.shape)
trace_model = torch.jit.trace(model, example_input_tensor)
torch.jit.save(trace_model, "vit_model.pt")
print("Model saved as vit_model.pt")


# Convert .pt model directly to Core ML (.mlpackage) format
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



