import base64
import sys
import os
import json
from adpipsvcfuncs import publish_to_pubsub, load_current_pipeline_data, save_current_pipeline_data
from adpipsvcfuncs import fetch_gcp_secret, load_valid_json
from adpipwfwconst import MSG_TYPE
from adpipwfwconst import PIPELINE_TOPICS as TOPICS
import logging
import requests
import tensorflow as tf
from build_ffn_configured import build_flexible_model
from load_features import main_training_process

logger = logging.getLogger('batch_logger')
# Trace if the logger is inheriting anything from its parent
if logger.parent:
    logger.debug(f"Batch logger parent: {logger.parent.name}")
    print(f"Batch logger parent: {logger.parent.name}")
else:
    logger.debug("Batch logger has no parent")
    print("Batch logger has no parent") 

# Ensure no handlers are inherited from the root logger
logger.handlers.clear()

logger.setLevel(logging.DEBUG)  # Capture DEBUG, INFO, WARNING, ERROR, CRITICAL
if not logger.handlers:
    # Create console handler and set its log level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # Add the handler to the root logger
    logger.addHandler(ch)

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)  # Capture DEBUG, INFO, WARNING, ERROR, CRITICAL

# Function to load data from the csv file located in the GCS bucket to the DataFrame
def load_data_from_gcs_bucket(bucket_name: str, file_name: str) -> dict:
    try:
        url = f"https://storage.googleapis.com/{bucket_name}/{file_name}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            logger.error(f"Failed to load data from the GCS bucket: {bucket_name}, file: {file_name}")
            return None
    except Exception as e:
        logger.error(f"Failed to load data from the GCS bucket: {bucket_name}, file: {file_name}, error: {e}")
        return None

# Function to get the FFN model configuration for a given pipeline_id
def adaptive_pipeline_get_model(pipeline_id: str) -> dict:
    
    pipeline_data = load_current_pipeline_data(pipeline_id)
    if pipeline_data is None:
        logger.error(f"Failed to load pipeline data for pipeline_id: {pipeline_id}")
        return None
    
    if 'current_configuration' not in pipeline_data:
        logger.error(f"current_configuration not found in pipeline data for pipeline_id: {pipeline_id}")
        return None
    
    model_config = pipeline_data.get('current_configuration')

    train_features_tensor, train_output_tensor, test_features_tensor, test_output_tensor = main_training_process()

    # Create a Keras input layer using the shape of the train_features_tensor
    input_tensor = tf.keras.Input(shape=train_features_tensor.shape[1:])

    hidden_layers_model = build_flexible_model(input_tensor, model_config)
    logger.error(f"DEBUG MODE break for pipeline_id: {pipeline_id}")
    logger.debug(f"DEBUG MODE resulting model for pipeline_id: {pipeline_id}:")
    hidden_layers_model.summary()

    #finalizing the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config['cfg']['lr'])
    logger.debug(f"Optimizer: {optimizer}")
    #model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    hidden_layers_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logger.debug(f"Model compiled")
    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=model_config['cfg']['lf'], patience=model_config['cfg']['lp'], 
        verbose=1, mode='auto', min_delta=model_config['cfg']['md'], cooldown=model_config['cfg']['cd'], min_lr=model_config['cfg']['mlr'])
    logger.debug(f"ReduceLR callback created")
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_config['cfg']['esp'], verbose=1, mode='auto', restire_best_weights=True)
    logger.debug(f"EarlyStop callback created")
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    logger.debug(f"Checkpoint callback created")
    history = hidden_layers_model.fit(train_features_tensor, train_output_tensor, batch_size=model_config['cfg']['bs'], epochs=model_config['cfg']['ep'],
        validation_data=(test_features_tensor, test_output_tensor), callbacks=[reduceLR, earlyStop, checkpoint], verbose=1)
    logger.debug(f"Model training completed")
    best_model = tf.keras.models.load_model('best_model.h5')
    logger.debug(f"Best model loaded")
    evaluation = best_model.evaluate(test_features_tensor, test_output_tensor)
    logger.debug(f"Model evaluation completed")
    accuracy = evaluation[1]
    loss = evaluation[0]
    logger.debug(f"Model evaluation: accuracy: {accuracy}, loss: {loss}")
    if accuracy > 0.9:
        logger.debug(f"Model training completed. Accuracy is more than 90%. Accuracy: {accuracy}, Loss: {loss}")
        message_data = {
            "pipeline_id": pipeline_id,
            "status": 10001,
            "accuracy": accuracy,
            "loss": loss
        }
    else:
        logger.debug(f"Model training completed. Accuracy is less than 90%. Accuracy: {accuracy}, Loss: {loss}")
        message_data = {
            "pipeline_id": pipeline_id,
            "status": 10002,
            "accuracy": accuracy,
            "loss": loss
        }
    
    if not publish_to_pubsub(TOPICS.WORKFLOW_TOPIC.value, message_data):
        logger.error("Failed to publish the message to the Pub/Sub topic")
        return False
    else:
        return True    


# Function that is triggered by a cloud function to process the batch data    
def train_model():
    # Trace if the logger is inheriting anything from its parent
    if logger.parent:
        logger.debug(f"Batch logger parent: {logger.parent.name}")
        print(f"Batch logger parent: {logger.parent.name}")
    else:
        logger.debug("Batch logger has no parent")
        print("Batch logger has no parent") 
    
    # Example of a very basic model training logic
    logger.debug("Starting model training...")

    # Debugging logs submission
    print("Testing print to stdout")
    sys.stdout.write("Testing sys.stdout.write\n")



    logger.debug("Testing DEBUG log")
    logger.info("Testing INFO log")
    logger.warning("Testing WARNING log")
    logger.error("Testing ERROR log")
    logger.critical("Testing CRITICAL log")    

    logger.debug("Loading pipeline data...")
    pipeline_id = os.getenv('PIPELINE_ID')

    try:
        pipeline_id = str(pipeline_id)
        if pipeline_id is None:
            logger.error("pipeline_id is not set")
        else:
            logger.debug(f"pipeline_id: {pipeline_id}")
            adaptive_pipeline_get_model(pipeline_id)
    except Exception as e:
        logger.error(f"Failed to load pipeline data. Error: {e}")       
    
    # Dummy model training process
    model_result = 2 + 2
    logger.debug(f"Model training completed. Result: {model_result}")
    # if dummy_pub_sub_message():
    #     logger.debug(f"Model training completed. Result: {model_result}")
    # else:
    #     logger.error("Failed to publish the message to the Pub/Sub topic")
    #     return
    logging.shutdown()

if __name__ == "__main__":
    train_model()