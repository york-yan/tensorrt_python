# Attention: This script requires ONNX and TensorRT to be installed. 

# First, convert pth to dynamic onnx, 
# and then convert dynamic onnx to static tensor to avoid insufficient memory of onnx when the size is larger.

# You can run it in Nvidia docker. 

import tensorrt as trt


def build_engine(onnx_file_path, engine_file_path, static_shape, precision_type=None):
    """
    Build a tensorRT engine from an dynamic onnx file with static shape in fp32 or fp16 precision.

    Args:
        onnx_path (str): The path to the ONNX file.
        engine_path (str): The path where the engine will be saved.
        static_shape (tuple): The static shape of the input tensor.
    Return:
        engine (trt.ICudaEngine): The engine.
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    if precision_type == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        print('WARNING: FP32 is used by default.')
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)  # Assuming one input. Adapt as necessary.
    profile.set_shape(input_tensor.name, min=static_shape, opt=static_shape, max=static_shape)
    config.add_optimization_profile(profile)

    engine = builder.build_engine(network, config)
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())
    return engine


class Calibrator(trt.IInt8Calibrator2):
    
 