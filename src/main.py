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
from load_features import load_features, save_data_to_gcs

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
db_schema = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "version": {
            "type": "string"
        },
        "pipeline_id": {
            "type": "string"
        },
        "status": {
            "type": "number",
        },
        "current_hidden_layers_ct": {
            "type": ["number", "null"]
        },
        "current_configuration": {
            "type": ["string", "null"]
        },
        "hidden_layers_configs": {
            "type": ["array", "null"],
            "items": {
                "hidden_layers_ct": {
                    "type": "number"
                },
                "is_completed": {
                    "type": "boolean"
                },
                "MAX_accuracy": {
                    "type": "number"
                },
                "configurations": {
                    "type": "array",
                    "items": {
                        "configuration": {
                            "type": "string"
                        },
                        "accuracy": {
                            "type": "number"
                        }
                    }
                }
            }
        }
    },
    "required": ["version", "pipeline_id", "status"],
    "additionalProperties": False
}    


def save_model_configuration_and_publish_message(pipeline_data: dict, accuracy: float) -> bool:
    # if hidden_layers_configs does not exist in the pipeline_data, create it and create a variable that stores a reference to it
    if "hidden_layers_configs" not in pipeline_data:
        pipeline_data['hidden_layers_configs'] = []
    hidden_layers_configs_dict = pipeline_data['hidden_layers_configs']

    # Gets the current_hidden_layers_ct from the pipeline_data checks if it is not set (throws an error if it is not set), 
    if "current_hidden_layers_ct" not in pipeline_data:
        logger.error("current_hidden_layers_ct is not set")
        return False
    current_hidden_layers_ct = pipeline_data['current_hidden_layers_ct']
    current_configuration = pipeline_data['current_configuration']
    if not current_configuration:
        logger.error("current_configuration is not set")
        return False
    if not hidden_layers_configs_dict:
        logger.error("hidden_layers_configs_dict is not set")
        return False
    # checks if there is an entry in the hidden_layers_configs_dict with the same hidden_layers_ct value as the current_hidden_layers_ct
    # if there is no entry, it creates a new entry in the hidden_layers_configs_dict with the current_configuration and accuracy values
    # if there is an entry, it updates the entry by adding the current_configuration and accuracy values to the configurations array

    for hidden_layers_config in hidden_layers_configs_dict:
        if hidden_layers_config['hidden_layers_ct'] == current_hidden_layers_ct:
            hidden_layers_config['configurations'].append({"configuration": current_configuration, "accuracy": accuracy})
            if accuracy > hidden_layers_config['MAX_accuracy']:
                hidden_layers_config['MAX_accuracy'] = accuracy
            logger.debug(f"Updated hidden_layers_config: {hidden_layers_config}")
            break
    else:
        hidden_layers_configs_dict.append(
            {
                "hidden_layers_ct": current_hidden_layers_ct, 
                "is_completed": False, 
                "MAX_accuracy": accuracy, 
                "configurations": [{
                    "configuration": current_configuration, "accuracy": accuracy}
                    ]})
        logger.debug(f"Added new hidden_layers_config: {hidden_layers_configs_dict[-1]}")



    #At the first run of the pipeline, the current_hidden_layers_ct is not set, so we set it to 1
    # if "current_hidden_layers_ct" not in pipeline_data:
    #     pipeline_data['current_hidden_layers_ct'] = 1
    pipeline_data['status'] = MSG_TYPE.START_MODEL_CONFIGURATION.value
    # API call to save the configuration
    save_current_pipeline_data(pipeline_data)    
    logger.debug(f"Pipeline data saved: {pipeline_data}")
    
    
    # pub_message_data = {
    # "pipeline_id": pipeline_data['pipeline_id'],
    # "status": MSG_TYPE.GENERATE_NEW_MODEL.value,
    # "current_configuration": pipeline_data['current_configuration']
    # }
    # publish_to_pubsub(TOPICS.WORKFLOW_TOPIC.value, pub_message_data)   
    # logger.debug(f"Publishing message to topic: {TOPICS.WORKFLOW_TOPIC.value} with data: {pub_message_data}")

# Function to get the FFN model configuration for a given pipeline_id
def adaptive_pipeline_get_model(pipeline_id: str) -> dict:
    
    pipeline_data = load_current_pipeline_data(pipeline_id)
    if pipeline_data is None:
        logger.error(f"Failed to load pipeline data for pipeline_id: {pipeline_id}")
        return None
    
    if 'current_configuration' not in pipeline_data:
        logger.error(f"current_configuration not found in pipeline data for pipeline_id: {pipeline_id}")
        return None
    
    logger.debug(f"Model configuration for pipeline_id: {pipeline_id}: {pipeline_data}")
    
    #model_config = json.loads(pipeline_data['current_configuration'])    
    model_config = pipeline_data['current_configuration']
    logger.debug(f"Model configuration for pipeline_id: {pipeline_id}: {model_config}")

    # Preparing the input and output layers
    train_features_tensor, train_output_tensor, test_features_tensor, test_output_tensor = load_features(False)
    logger.debug(f"Train features tensor shape: {train_features_tensor.shape}")
    logger.debug(f"Train output tensor shape: {train_output_tensor.shape}")
    logger.debug(f"Test features tensor shape: {test_features_tensor.shape}")
    logger.debug(f"Test output tensor shape: {test_output_tensor.shape}")
    # Create a Keras input layer using the shape of the train_features_tensor
    input_tensor = tf.keras.Input(shape=train_features_tensor.shape[1:])

    # Preparing hidden layers model
    hidden_layers_model = build_flexible_model(input_tensor, model_config)
    logger.error(f"DEBUG MODE break for pipeline_id: {pipeline_id}")
    logger.debug(f"DEBUG MODE resulting model for pipeline_id: {pipeline_id}:")
    hidden_layers_model.summary()

    #Training the model
    model_config_json = json.loads(model_config)
    logger.debug(f"Starting model training...")
    #finalizing the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=model_config_json['cfg']['lr'])
    logger.debug(f"Optimizer: {optimizer}")
    #model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    hidden_layers_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    logger.debug(f"Model compiled")
    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=model_config_json['cfg']['lf'], patience=model_config_json['cfg']['lp'], 
        verbose=1, mode='auto', min_delta=model_config_json['cfg']['md'], cooldown=model_config_json['cfg']['cd'], min_lr=model_config_json['cfg']['mlr'])
    logger.debug(f"ReduceLR callback created")
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=model_config_json['cfg']['esp'], verbose=1, mode='auto', restore_best_weights=True)
    logger.debug(f"EarlyStop callback created")
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    logger.debug(f"Checkpoint callback created")
    
    history = hidden_layers_model.fit(train_features_tensor, train_output_tensor, batch_size=model_config_json['cfg']['bs'], epochs=model_config_json['cfg']['ep'],
        validation_data=(test_features_tensor, test_output_tensor), callbacks=[reduceLR, earlyStop, checkpoint], verbose=1)
    
    logger.debug(f"Model training completed")
    best_model = tf.keras.models.load_model('best_model.keras')
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
        # updating the pipeline data with the model training result and logging the model training result as a part of the pipeline data
        # 

    
    save_data_to_gcs(pipeline_id, best_model)
    logger.debug(f"Model saved to GCS bucket")

    save_model_configuration_and_publish_message(pipeline_data, accuracy)
    
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

    pipeline_id = str(pipeline_id)
    adaptive_pipeline_get_model(pipeline_id)

    # try:
    #     pipeline_id = str(pipeline_id)
    #     if pipeline_id is None:
    #         logger.error("pipeline_id is not set")
    #     else:
    #         logger.debug(f"pipeline_id: {pipeline_id}")
    #         adaptive_pipeline_get_model(pipeline_id)
    # except Exception as e:
    #     logger.error(f"Failed to load pipeline data. Error: {e}")       
    
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