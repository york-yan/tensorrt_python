import torch 
import onnxruntime as ort
import numpy as np


def exoport_onnx_model(model,dummy_input,onnx_output_path,input_names=['input'],output_names=['output'],dynamic_axes=None):
    """
    Exports the given PyTorch model to an ONNX file.

    Args:
        model (torch.nn.Module): The PyTorch model to be exported.
        dummy_input (torch.Tensor): A dummy input tensor that matches the shape and dtype of the model's input.
        onnx_output_path (str): The path where the ONNX file will be saved.
        input_names (List[str], optional): The names of the model's input tensors. Defaults to ['input'].
        output_names (List[str], optional): The names of the model's output tensors. Defaults to ['output'].
        dynamic_axes (Dict[str, Dict[int, str]], optional): A dictionary specifying the dynamic axes of the model's input tensors. Defaults to None.
    """
    torch.onnx.export(model,dummy_input,onnx_output_path,input_names=input_names,output_names=output_names,dynamic_axes=dynamic_axes)

def onnx_validate(onnx_path):
    """
    Validates the ONNX model using the onnxruntime package.

    Args:
        onnx_path (str): The path to the ONNX model.
    """
    provider=ort.get_available_providers()[1 if ort.get_device() == 'GPU' else 0]
    print('device:', provider)
    model=ort.InferenceSession(onnx_path,providers=[provider])
    for node_list in model.get_inputs(), model.get_outputs():
        for node in node_list:
            attr = {'name': node.name,
                    'shape': node.shape,
                    'type': node.type}
            print(attr)
        print('-' * 80)
        