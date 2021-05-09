import pandas as pd
import click
import yaml
import joblib
import logging.config
from cloudpickle import load
from data import read_csv, get_train_test_data
from model_pipeline import DataProcessingPipeline, Classifier, get_classification_report
from enities import read_training_pipeline_params

APPLICATION_NAME = 'homework1'
logger = logging.getLogger(APPLICATION_NAME)


def setup_logging(path):
    with open(path) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def train_p(params):
    data = read_csv(params.input_data_path)
    logger.info(f"data loaded, DF shape{data.shape}")
    train_df, test_df = get_train_test_data(data, params.split_params)
    logger.info(f"train_df shape is {train_df.shape}")
    logger.info(f"test_df shape is {test_df.shape}")
    data_preprocessing_pipeline = DataProcessingPipeline(params.feature_params.categorical_features,
                                                         params.feature_params.numerical_features)
    logger.info("Data preprocessing pipeline initiated")
    classifier = Classifier(params.classifier_params, params.model_type)
    logger.info("Classifier initiated")
    data_preprocessing_pipeline.fit(train_df)
    logger.info("Data preprocessing pipeline fitted")
    transformed_train = data_preprocessing_pipeline.transform(train_df)
    logger.info(f"transfromed train data shape {transformed_train.shape}")
    classifier.fit(transformed_train, train_df[params.feature_params.target].values)
    logger.info("classifier fitted")
    transformed_test = data_preprocessing_pipeline.transform(test_df)
    logger.info(f"tranfromed test data shape {transformed_test.shape}")
    y_pred = classifier.predict(transformed_test)
    report_dict = get_classification_report(test_df[params.feature_params.target].values, y_pred)
    logger.info(f"Classification report {report_dict}")
    classifier.dump_model(params.output_model_path)
    logger.info(f"Classifier dumped {params.output_model_path}")
    data_preprocessing_pipeline.dump_preprocessor(params.output_data_preprocessor_path)
    logger.info(f"Data preprocessor dumped {params.output_data_preprocessor_path}")

def validate_p(params):
    data = read_csv(params.input_data_for_validation)
    logger.info(f"Validation df loaded, shape:{data.shape}")
    classifier = Classifier(params.classifier_params, params.model_type)
    classifier.model = joblib.load(params.output_model_path)
    logger.info(f"Model type {params.model_type} loaded from {params.output_model_path}")
    data_preprocessing_pipeline = DataProcessingPipeline(params.feature_params.categorical_features,
                                                         params.feature_params.numerical_features)
    with open(params.output_data_preprocessor_path, 'rb') as f:
        data_preprocessing_pipeline.pipeline = load(f)
        logger.info(f"Data preprocessor loaded from {params.output_data_preprocessor_path}")
    transformed_data = data_preprocessing_pipeline.transform(data)
    logger.info(f"data for validation preprocessed, shape: {transformed_data.shape}")
    y_pred = classifier.predict(transformed_data)
    logger.info(f"Predicts prepared")
    pd.DataFrame(y_pred, columns=["target"]).to_csv(params.predicts_path)
    logger.info(f"Predicts dumped to {params.predicts_path}")



@click.command(name="train_p")
@click.argument("config_path")
@click.argument("train_or_validate")
def train_pipeline_command(config_path: str, train_or_validate: str):
    params = read_training_pipeline_params(config_path)
    setup_logging(params.logger_config)
    if train_or_validate == "train":
        logger.warning("App initiated in train mode")
        train_pipeline(params)
    elif train_or_validate == "validate":
        logger.warning("App initiated in validation mode")
        validate_pipeline(params)
    else:
        logger.warning("Incorrect train or valiidation mode")


if __name__ == "__main__":
    train_pipeline_command()