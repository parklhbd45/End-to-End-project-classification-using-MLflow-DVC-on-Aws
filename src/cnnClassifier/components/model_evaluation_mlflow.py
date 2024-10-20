import tensorflow as tf
from pathlib import Path
import mlflow
from mlflow import MlflowException
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories,save_json


class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):

        try:
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # ใช้ชื่อ experiment แบบ hardcode หรือใช้ชื่อโปรเจกต์ของคุณ
            experiment_name = "ChestClassifier_Experiment"

            # Check if the experiment exists, if not create it
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id

            # สร้างชื่อ run โดยใช้เวลาปัจจุบัน
            from datetime import datetime
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
                mlflow.log_params(self.config.all_params)
                mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
                else:
                    mlflow.keras.log_model(self.model, "model")

                print("Model logged in MLflow successfully")

        except MlflowException as e:
            print(f"Error logging to MLflow: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
