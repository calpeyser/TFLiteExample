tflite_convert \
	--allow_custom_ops \
	--output_file=/tmp/models/model.tflite \
#	--output_format=GRAPHVIZ_DOT \
	--saved_model_dir=/tmp/models \
    --target_ops=TFLITE_BUILTINS \
	--input_arrays=Placeholder \
	--input_shapes=1 \
	--output_arrays=Sin \
	--output_shapes=1
