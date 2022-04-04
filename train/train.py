"""
Training code
"""
import torch
import torchvision

# An instance of your model.
model = torchvision.models.densenet201(pretrained=True)

model.eval()

torch.save(model, "../hub/model.pt")

# Export the model
torch.onnx.export(
    model,                              # model being run
    torch.rand(1, 3, 224, 224),         # model input (or a tuple for multiple inputs)
    "../hub/model.onnx",                # where to save the model (can be a file or file-like object)
    export_params=True,                 # store the trained parameter weights inside the model file
    opset_version=14,                   # the ONNX version to export the model to
    do_constant_folding=True,           # whether to execute constant folding for optimization
    input_names=['input'],            # the model's input names
    output_names=['output'],          # the model's output names
)

# tracing and scripting a module ? ()
