from hailo_sdk_client import ClientRunner

model_name = 'ei-car-yolov5s'
onnx_path = f'{model_name}.onnx'
chosen_hw_arch = 'hailo8l'

runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(
    onnx_path,
    model_name,
    start_node_names=['images'],
    end_node_names=['/model.24/m.0/Conv', '/model.24/m.2/Conv', '/model.24/m.1/Conv'],
    net_input_shapes={'images': [1, 3, 640, 640]})

runner.save_har(f'{model_name}.har')

