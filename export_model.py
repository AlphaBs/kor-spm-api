from tf_keras.models import load_model
import tensorflow as tf

# Load the Keras model
model = load_model('cnn-spm.keras')

# Define the serving signature
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 200], dtype=tf.int32, name='input')])
def serving_fn(input_tensor):
    return {'output': model(input_tensor)}

# Save the model in SavedModel format
tf.saved_model.save(
    model,
    './models/cnn-spm/1',
    signatures={'serving_default': serving_fn}
)

print("Model saved to ./models/cnn-spm/1")
